"""Step action handlers.

Each handler takes an :class:`ActionContext` plus a :class:`WorkflowStep` and
returns a :class:`StepOutcome`; the runner wraps handlers with timing, JSONL
logging and failure screenshots.

Dry-run contract: handlers must not move the mouse, click, type, paste or
press hotkeys when ``ctx.dry_run`` is True. Locating targets (screenshots +
matching) is allowed — that is the whole point of a dry run.
"""

from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..locator import Locator, MatchResult, build_locator, save_debug_image
from ..models import (
    STATUS_FAILURE,
    STATUS_SUCCESS,
    Target,
    WorkflowStep,
)
from ..permissions import is_macos
from ..screen import ScreenSource, display_scale
from ..storage import ProfileStore
from .controls import InputController


class RunCancelled(Exception):
    """Raised inside handlers when the user cancels the run."""


@dataclass
class StepOutcome:
    status: str
    message: str = ""
    target_id: str | None = None
    confidence: float | None = None
    location: tuple[int, int] | None = None  # logical screen coords
    artifacts: list[str] = field(default_factory=list)


@dataclass
class ActionContext:
    store: ProfileStore
    screen: ScreenSource
    controller: InputController
    run_dir: Path
    dry_run: bool = False
    save_debug: bool = False
    poll_interval: float = 0.5
    pre_click_pause: float = 0.15
    cancel_event: threading.Event = field(default_factory=threading.Event)
    variables: dict[str, Any] = field(default_factory=dict)
    _locator_cache: dict[str, Locator] = field(default_factory=dict)

    def check_cancel(self) -> None:
        if self.cancel_event.is_set():
            raise RunCancelled()

    def sleep(self, seconds: float) -> None:
        """Sleep, waking immediately (and raising) if the run is cancelled."""
        if seconds > 0 and self.cancel_event.wait(timeout=seconds):
            raise RunCancelled()
        self.check_cancel()

    def locator_for(self, target: Target) -> Locator:
        key = f"{target.locator.method}:{sorted(target.locator.params.items())!r}"
        if key not in self._locator_cache:
            self._locator_cache[key] = build_locator(target.locator)
        return self._locator_cache[key]


@dataclass
class _Sighting:
    """One locate attempt: the match plus where it lands in logical coords."""

    match: MatchResult
    point: tuple[int, int] | None
    screenshot: np.ndarray


def _locate_once(ctx: ActionContext, target: Target) -> _Sighting:
    screenshot = ctx.screen.screenshot()
    template = ctx.store.load_template(target)
    match = ctx.locator_for(target).find_in_image(screenshot, template, target.min_confidence)
    point = None
    if match.center is not None:
        scale = display_scale(screenshot, ctx.screen.logical_size())
        dx, dy = target.resolve_click_offset()
        point = (round(match.center[0] / scale) + dx, round(match.center[1] / scale) + dy)
    return _Sighting(match=match, point=point, screenshot=screenshot)


def _maybe_save_debug(ctx: ActionContext, target: Target, sighting: _Sighting) -> list[str]:
    if not ctx.save_debug:
        return []
    filename = f"debug_{target.id}.png"
    try:
        save_debug_image(ctx.run_dir / filename, sighting.screenshot, sighting.match)
    except Exception:
        return []
    return [filename]


def _locate_with_timeout(
    ctx: ActionContext, target: Target, timeout_seconds: float
) -> tuple[_Sighting, float]:
    """Poll for a target until found or the deadline passes.

    Returns the final (best-effort) sighting and the best confidence seen.
    """
    deadline = time.monotonic() + timeout_seconds
    best_confidence = 0.0
    while True:
        ctx.check_cancel()
        sighting = _locate_once(ctx, target)
        best_confidence = max(best_confidence, sighting.match.confidence)
        if sighting.match.found:
            return sighting, best_confidence
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return sighting, best_confidence
        ctx.sleep(min(ctx.poll_interval, remaining))


def _not_found_message(target: Target, best_confidence: float, timeout_seconds: float) -> str:
    return (
        f"Target '{target.id}' not found within {timeout_seconds:g}s "
        f"(best confidence {best_confidence:.3f}, min {target.min_confidence:.2f}). "
        "Try re-capturing the template or lowering min_confidence."
    )


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def action_wait_seconds(ctx: ActionContext, step: WorkflowStep) -> StepOutcome:
    seconds = float(step.seconds or 0)
    if ctx.dry_run:
        return StepOutcome(STATUS_SUCCESS, f"Dry run: would wait {seconds:g}s")
    ctx.sleep(seconds)
    return StepOutcome(STATUS_SUCCESS, f"Waited {seconds:g}s")


def action_find_target(ctx: ActionContext, step: WorkflowStep) -> StepOutcome:
    target = ctx.store.load_target(step.target_id)
    sighting = _locate_once(ctx, target)
    artifacts = _maybe_save_debug(ctx, target, sighting)
    if sighting.match.found:
        x, y = sighting.point
        return StepOutcome(
            STATUS_SUCCESS,
            f"Found target '{target.id}' at x={x}, y={y} ({sighting.match.message})",
            target_id=target.id,
            confidence=sighting.match.confidence,
            location=sighting.point,
            artifacts=artifacts,
        )
    return StepOutcome(
        STATUS_FAILURE,
        f"Target '{target.id}' not found ({sighting.match.message})",
        target_id=target.id,
        confidence=sighting.match.confidence,
        artifacts=artifacts,
    )


def action_wait_for_target(ctx: ActionContext, step: WorkflowStep) -> StepOutcome:
    target = ctx.store.load_target(step.target_id)
    started = time.monotonic()
    sighting, best_confidence = _locate_with_timeout(ctx, target, step.timeout_seconds)
    artifacts = _maybe_save_debug(ctx, target, sighting)
    if sighting.match.found:
        elapsed = time.monotonic() - started
        return StepOutcome(
            STATUS_SUCCESS,
            f"Target '{target.id}' appeared after {elapsed:.1f}s",
            target_id=target.id,
            confidence=sighting.match.confidence,
            location=sighting.point,
            artifacts=artifacts,
        )
    return StepOutcome(
        STATUS_FAILURE,
        _not_found_message(target, best_confidence, step.timeout_seconds),
        target_id=target.id,
        confidence=best_confidence,
        artifacts=artifacts,
    )


def action_wait_for_any_target(ctx: ActionContext, step: WorkflowStep) -> StepOutcome:
    targets = [ctx.store.load_target(target_id) for target_id in step.target_ids]
    deadline = time.monotonic() + step.timeout_seconds
    started = time.monotonic()
    best: dict[str, float] = {t.id: 0.0 for t in targets}
    while True:
        ctx.check_cancel()
        for target in targets:
            sighting = _locate_once(ctx, target)
            best[target.id] = max(best[target.id], sighting.match.confidence)
            if sighting.match.found:
                elapsed = time.monotonic() - started
                artifacts = _maybe_save_debug(ctx, target, sighting)
                return StepOutcome(
                    STATUS_SUCCESS,
                    f"Target '{target.id}' appeared after {elapsed:.1f}s",
                    target_id=target.id,
                    confidence=sighting.match.confidence,
                    location=sighting.point,
                    artifacts=artifacts,
                )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            summary = ", ".join(f"{tid}={conf:.3f}" for tid, conf in best.items())
            return StepOutcome(
                STATUS_FAILURE,
                f"None of the targets appeared within {step.timeout_seconds:g}s "
                f"(best confidences: {summary}).",
                confidence=max(best.values(), default=0.0),
            )
        ctx.sleep(min(ctx.poll_interval, remaining))


def action_click_target(ctx: ActionContext, step: WorkflowStep) -> StepOutcome:
    target = ctx.store.load_target(step.target_id)
    sighting, best_confidence = _locate_with_timeout(ctx, target, step.timeout_seconds)
    artifacts = _maybe_save_debug(ctx, target, sighting)
    if not sighting.match.found:
        # Guardrail: never click below the target's min_confidence.
        return StepOutcome(
            STATUS_FAILURE,
            _not_found_message(target, best_confidence, step.timeout_seconds),
            target_id=target.id,
            confidence=best_confidence,
            artifacts=artifacts,
        )
    x, y = sighting.point
    if ctx.dry_run:
        return StepOutcome(
            STATUS_SUCCESS,
            f"Dry run: would click target '{target.id}' at x={x}, y={y}",
            target_id=target.id,
            confidence=sighting.match.confidence,
            location=sighting.point,
            artifacts=artifacts,
        )
    # Move first and pause briefly so the user can see (and abort) the click.
    ctx.controller.move_to(x, y)
    ctx.sleep(float(step.params.get("pre_click_pause", ctx.pre_click_pause)))
    ctx.controller.click(x, y)
    return StepOutcome(
        STATUS_SUCCESS,
        f"Clicked target at x={x}, y={y}",
        target_id=target.id,
        confidence=sighting.match.confidence,
        location=sighting.point,
        artifacts=artifacts,
    )


def action_paste_text(ctx: ActionContext, step: WorkflowStep) -> StepOutcome:
    text = step.text or ""
    if ctx.dry_run:
        return StepOutcome(STATUS_SUCCESS, f"Dry run: would paste {len(text)} characters")
    ctx.controller.paste_text(text)
    return StepOutcome(STATUS_SUCCESS, f"Pasted {len(text)} characters")


def action_copy_clipboard(ctx: ActionContext, step: WorkflowStep) -> StepOutcome:
    if ctx.dry_run:
        return StepOutcome(
            STATUS_SUCCESS, "Dry run: would copy the current selection to the clipboard"
        )
    text = ctx.controller.copy_selection()
    ctx.variables["clipboard"] = text
    preview = text[:80].replace("\n", "\\n")
    suffix = "…" if len(text) > 80 else ""
    return StepOutcome(
        STATUS_SUCCESS, f"Copied {len(text)} characters to clipboard: '{preview}{suffix}'"
    )


def action_hotkey(ctx: ActionContext, step: WorkflowStep) -> StepOutcome:
    combo = "+".join(step.keys)
    if ctx.dry_run:
        return StepOutcome(STATUS_SUCCESS, f"Dry run: would press {combo}")
    ctx.controller.hotkey(*step.keys)
    return StepOutcome(STATUS_SUCCESS, f"Pressed {combo}")


def action_screenshot(ctx: ActionContext, step: WorkflowStep) -> StepOutcome:
    screenshot = ctx.screen.screenshot()
    filename = f"screenshot_{step.id}.png"
    if not cv2.imwrite(str(ctx.run_dir / filename), screenshot):
        return StepOutcome(STATUS_FAILURE, f"Could not save screenshot to {filename}")
    return StepOutcome(STATUS_SUCCESS, f"Saved screenshot {filename}", artifacts=[filename])


def action_notify(ctx: ActionContext, step: WorkflowStep) -> StepOutcome:
    message = step.message or "Notification from workflow"
    if not ctx.dry_run and step.params.get("system_notification") and is_macos():
        # Best-effort macOS notification banner; the log line is the real record.
        try:
            import json as _json

            script = f'display notification {_json.dumps(message)} with title "Local RPA Studio"'
            subprocess.run(["osascript", "-e", script], capture_output=True, timeout=5)
        except Exception:
            pass
    return StepOutcome(STATUS_SUCCESS, message)


def action_stop(ctx: ActionContext, step: WorkflowStep) -> StepOutcome:
    return StepOutcome(STATUS_SUCCESS, "Workflow stopped by 'stop' step")


ACTION_HANDLERS = {
    "wait_seconds": action_wait_seconds,
    "find_target": action_find_target,
    "wait_for_target": action_wait_for_target,
    "wait_for_any_target": action_wait_for_any_target,
    "click_target": action_click_target,
    "paste_text": action_paste_text,
    "copy_clipboard": action_copy_clipboard,
    "hotkey": action_hotkey,
    "screenshot": action_screenshot,
    "notify": action_notify,
    "stop": action_stop,
}
