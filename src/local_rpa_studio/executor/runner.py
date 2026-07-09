"""Workflow runner: executes steps, logs JSONL events, saves failure artifacts.

Supports normal and dry runs, a pre-run countdown so the user can focus the
target app, per-step timeouts (enforced inside the wait/click handlers),
stop-on-failure, and cooperative cancellation via a threading.Event (the UI's
Cancel button simply sets it).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import cv2

from ..locator import LocatorError
from ..logging_utils import RunLogger
from ..models import (
    STATUS_CANCELLED,
    STATUS_FAILURE,
    STATUS_SKIPPED,
    STATUS_SUCCESS,
    RunResult,
    StepResult,
    Workflow,
    WorkflowStep,
    now_iso,
    referenced_target_ids,
    validate_workflow,
)
from ..screen import RealScreen, ScreenCaptureError, ScreenSource
from ..storage import ProfileStore, StorageError
from .actions import ACTION_HANDLERS, ActionContext, RunCancelled, StepOutcome
from .controls import InputControlError, InputController

DEFAULT_COUNTDOWN_SECONDS = 3.0


class WorkflowRunner:
    def __init__(
        self,
        store: ProfileStore,
        workflow: Workflow,
        *,
        dry_run: bool = False,
        screen: ScreenSource | None = None,
        controller: InputController | None = None,
        countdown_seconds: float | None = None,
        save_debug: bool | None = None,
        poll_interval: float = 0.5,
        pre_click_pause: float = 0.15,
        on_event: Callable[[dict], None] | None = None,
        cancel_event: threading.Event | None = None,
    ):
        self.store = store
        self.workflow = workflow
        self.dry_run = dry_run
        self.screen = screen if screen is not None else RealScreen()
        self.controller = controller if controller is not None else InputController()
        if countdown_seconds is None:
            countdown_seconds = (
                0.0
                if dry_run
                else (
                    workflow.countdown_seconds
                    if workflow.countdown_seconds is not None
                    else DEFAULT_COUNTDOWN_SECONDS
                )
            )
        self.countdown_seconds = countdown_seconds
        # Debug images are cheap and most useful when previewing matches.
        self.save_debug = dry_run if save_debug is None else save_debug
        self.poll_interval = poll_interval
        self.pre_click_pause = pre_click_pause
        self.on_event = on_event
        self.cancel_event = cancel_event or threading.Event()
        self.run_dir = None  # set once run() starts

    def cancel(self) -> None:
        self.cancel_event.set()

    # ------------------------------------------------------------------

    def _validate(self) -> list[str]:
        known_ids = set(self.store.list_target_ids())
        errors = validate_workflow(self.workflow, known_target_ids=known_ids)
        for target_id in sorted(referenced_target_ids(self.workflow) & known_ids):
            try:
                target = self.store.load_target(target_id)
                target.resolve_click_offset()
                if not self.store.template_abspath(target).is_file():
                    errors.append(
                        f"target '{target_id}' has no template image "
                        f"({target.template_path}); re-capture it"
                    )
            except (StorageError, ValueError) as exc:
                errors.append(str(exc))
        return errors

    def run(self) -> RunResult:
        run_dir = self.store.new_run_dir()
        self.run_dir = run_dir
        logger = RunLogger(
            run_dir,
            workflow_id=self.workflow.id,
            profile_id=self.store.profile_id,
            dry_run=self.dry_run,
            echo=self.on_event,
        )
        result = RunResult(
            run_id=run_dir.name,
            workflow_id=self.workflow.id,
            profile_id=self.store.profile_id,
            dry_run=self.dry_run,
            started_at=now_iso(),
            run_dir=str(run_dir),
        )

        errors = self._validate()
        if errors:
            result.status = STATUS_FAILURE
            result.message = "Workflow validation failed: " + "; ".join(errors)
            logger.log("run_failed", message=result.message)
            result.finished_at = now_iso()
            return result

        mode = "DRY RUN" if self.dry_run else "run"
        logger.log(
            "run_started",
            message=(
                f"Starting {mode} of workflow '{self.workflow.display_name}' "
                f"({len(self.workflow.steps)} steps)"
            ),
        )

        ctx = ActionContext(
            store=self.store,
            screen=self.screen,
            controller=self.controller,
            run_dir=run_dir,
            dry_run=self.dry_run,
            save_debug=self.save_debug,
            poll_interval=self.poll_interval,
            pre_click_pause=self.pre_click_pause,
            cancel_event=self.cancel_event,
        )

        cancelled = self._countdown(ctx, logger)
        stopped_early = False
        remaining_steps: list[WorkflowStep] = list(self.workflow.steps)

        if not cancelled:
            for index, step in enumerate(self.workflow.steps):
                remaining_steps = list(self.workflow.steps[index + 1 :])
                if self.cancel_event.is_set():
                    step_result = StepResult(
                        step_id=step.id,
                        action=step.action,
                        status=STATUS_CANCELLED,
                        started_at=now_iso(),
                        finished_at=now_iso(),
                        message="Run cancelled by user",
                    )
                    self._log_step_finished(logger, step_result, duration=0.0)
                    result.step_results.append(step_result)
                    cancelled = True
                    break

                step_result = self._run_step(ctx, step, logger)
                result.step_results.append(step_result)

                if step_result.status == STATUS_CANCELLED:
                    cancelled = True
                    break
                if step.action == "stop" and step_result.status == STATUS_SUCCESS:
                    stopped_early = True
                    break
                if step_result.status == STATUS_FAILURE:
                    if step.continue_on_failure:
                        logger.log(
                            "info",
                            message=(
                                f"Step '{step.id}' failed but has "
                                "continue_on_failure: true — continuing"
                            ),
                        )
                        continue
                    if self.workflow.stop_on_failure:
                        stopped_early = True
                        break
            else:
                remaining_steps = []

        if remaining_steps and (cancelled or stopped_early):
            for step in remaining_steps:
                skipped = StepResult(
                    step_id=step.id,
                    action=step.action,
                    status=STATUS_SKIPPED,
                    started_at=now_iso(),
                    finished_at=now_iso(),
                    message="Skipped (run ended before this step)",
                )
                self._log_step_finished(logger, skipped, duration=0.0)
                result.step_results.append(skipped)

        failures = [r for r in result.step_results if r.status == STATUS_FAILURE]
        if cancelled:
            result.status = STATUS_CANCELLED
            result.message = "Run cancelled by user"
        elif failures:
            result.status = STATUS_FAILURE
            first = failures[0]
            result.message = f"Step '{first.step_id}' failed: {first.message}"
        else:
            result.status = STATUS_SUCCESS
            result.message = "All steps completed"

        result.finished_at = now_iso()
        counts = {
            status: sum(1 for r in result.step_results if r.status == status)
            for status in (STATUS_SUCCESS, STATUS_FAILURE, STATUS_SKIPPED, STATUS_CANCELLED)
        }
        logger.log(
            "run_finished",
            status=result.status,
            message=(
                f"Run finished: {result.status.upper()} — "
                f"{counts[STATUS_SUCCESS]} succeeded, {counts[STATUS_FAILURE]} failed, "
                f"{counts[STATUS_SKIPPED]} skipped. {result.message}"
            ),
        )
        return result

    # ------------------------------------------------------------------

    def _countdown(self, ctx: ActionContext, logger: RunLogger) -> bool:
        """Count down before touching anything. Returns True if cancelled."""
        try:
            for remaining in range(int(self.countdown_seconds), 0, -1):
                logger.log(
                    "countdown",
                    message=f"Starting in {remaining}s — focus the target app/window now…",
                )
                ctx.sleep(1.0)
        except RunCancelled:
            logger.log("info", message="Run cancelled during countdown")
            return True
        return False

    def _run_step(self, ctx: ActionContext, step: WorkflowStep, logger: RunLogger) -> StepResult:
        started_at = now_iso()
        start = time.monotonic()
        logger.log("step_started", step_id=step.id, action=step.action, target_id=step.target_id)

        try:
            handler = ACTION_HANDLERS[step.action]  # validated before the run
            outcome = handler(ctx, step)
        except RunCancelled:
            outcome = StepOutcome(STATUS_CANCELLED, "Run cancelled by user")
        except (StorageError, ScreenCaptureError, InputControlError, LocatorError) as exc:
            outcome = StepOutcome(STATUS_FAILURE, str(exc))
        except Exception as exc:  # keep the run log intact on unexpected bugs
            outcome = StepOutcome(STATUS_FAILURE, f"Unexpected error: {exc!r}")

        duration = time.monotonic() - start
        step_result = StepResult(
            step_id=step.id,
            action=step.action,
            status=outcome.status,
            started_at=started_at,
            finished_at=now_iso(),
            message=outcome.message,
            error=outcome.message if outcome.status == STATUS_FAILURE else None,
            target_id=outcome.target_id or step.target_id,
            confidence=outcome.confidence,
            location=outcome.location,
            artifacts=list(outcome.artifacts),
        )

        if step_result.status == STATUS_FAILURE:
            artifact = self._save_failure_screenshot(ctx, step)
            if artifact:
                step_result.artifacts.append(artifact)

        self._log_step_finished(logger, step_result, duration)
        return step_result

    def _save_failure_screenshot(self, ctx: ActionContext, step: WorkflowStep) -> str | None:
        filename = f"failure_{step.id}.png"
        try:
            screenshot = ctx.screen.screenshot()
            if cv2.imwrite(str(ctx.run_dir / filename), screenshot):
                return filename
        except Exception:
            pass
        return None

    @staticmethod
    def _log_step_finished(logger: RunLogger, result: StepResult, duration: float) -> None:
        logger.log(
            "step_finished",
            step_id=result.step_id,
            action=result.action,
            status=result.status,
            target_id=result.target_id,
            confidence=result.confidence,
            message=result.message,
            error=result.error,
            location=list(result.location) if result.location else None,
            artifacts=result.artifacts or None,
            duration_seconds=round(duration, 3),
        )
