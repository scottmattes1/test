"""Data models for profiles, targets, workflows, and run results.

Plain dataclasses with explicit to_dict/from_dict helpers so the YAML files on
disk stay compact and human-editable (fields equal to their defaults are
omitted when serialising; unknown keys are tolerated when loading).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

VALID_ACTIONS: frozenset[str] = frozenset(
    {
        "wait_seconds",
        "find_target",
        "wait_for_target",
        "wait_for_any_target",
        "click_target",
        "paste_text",
        "copy_clipboard",
        "hotkey",
        "screenshot",
        "notify",
        "stop",
    }
)

TARGET_KINDS: tuple[str, ...] = ("button", "label", "input", "icon", "region", "other")

STATUS_SUCCESS = "success"
STATUS_FAILURE = "failure"
STATUS_SKIPPED = "skipped"
STATUS_CANCELLED = "cancelled"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def slugify(text: str) -> str:
    """Turn arbitrary text into a safe id: lowercase, [a-z0-9_] only."""
    slug = re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")
    return slug or "item"


@dataclass
class LocatorStrategy:
    """How a target is located on screen.

    ``vision_template`` (OpenCV template matching) is the only method in the
    MVP; ``accessibility``, ``ocr`` and ``playwright`` are planned. ``params``
    carries method-specific options (e.g. ``scales`` for multi-scale matching).
    """

    method: str = "vision_template"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Target:
    """A visual landmark captured from the screen (button, label, field...)."""

    id: str
    display_name: str
    kind: str = "button"
    template_path: str = ""  # relative to the profile directory
    min_confidence: float = 0.88
    search_region: str = "active_screen"  # MVP: full primary screen only
    click_offset: str = "center"  # "center" or "dx,dy" (logical px from centre)
    locator: LocatorStrategy = field(default_factory=LocatorStrategy)
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)

    def resolve_click_offset(self) -> tuple[int, int]:
        """Return the (dx, dy) offset applied to the match centre before clicking."""
        raw = (self.click_offset or "center").strip().lower()
        if raw in ("", "center", "centre"):
            return (0, 0)
        parts = raw.split(",")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid click_offset {self.click_offset!r} on target {self.id!r}: "
                "use 'center' or 'dx,dy' (e.g. '10,-4')."
            )
        try:
            return (int(parts[0].strip()), int(parts[1].strip()))
        except ValueError as exc:
            raise ValueError(
                f"Invalid click_offset {self.click_offset!r} on target {self.id!r}: "
                "offsets must be integers."
            ) from exc


@dataclass
class WorkflowStep:
    id: str
    action: str
    target_id: str | None = None
    target_ids: list[str] = field(default_factory=list)  # wait_for_any_target
    text: str | None = None  # paste_text
    keys: list[str] = field(default_factory=list)  # hotkey
    seconds: float | None = None  # wait_seconds
    timeout_seconds: float = 10.0
    message: str | None = None  # notify
    continue_on_failure: bool = False
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Workflow:
    id: str
    display_name: str
    description: str = ""
    countdown_seconds: float | None = None  # None -> runner default
    stop_on_failure: bool = True
    steps: list[WorkflowStep] = field(default_factory=list)


@dataclass
class AppProfile:
    id: str
    display_name: str
    description: str = ""
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)


@dataclass
class StepResult:
    step_id: str
    action: str
    status: str
    started_at: str = ""
    finished_at: str = ""
    message: str = ""
    error: str | None = None
    target_id: str | None = None
    confidence: float | None = None
    location: tuple[int, int] | None = None  # logical screen coords
    artifacts: list[str] = field(default_factory=list)  # file names inside the run dir


@dataclass
class RunResult:
    run_id: str
    workflow_id: str
    profile_id: str
    dry_run: bool
    status: str = STATUS_SUCCESS
    started_at: str = ""
    finished_at: str = ""
    message: str = ""
    step_results: list[StepResult] = field(default_factory=list)
    run_dir: str = ""

    @property
    def ok(self) -> bool:
        return self.status == STATUS_SUCCESS


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def profile_to_dict(profile: AppProfile) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": profile.id,
        "display_name": profile.display_name,
        "created_at": profile.created_at,
        "updated_at": profile.updated_at,
    }
    if profile.description:
        data["description"] = profile.description
    return data


def profile_from_dict(data: dict[str, Any]) -> AppProfile:
    return AppProfile(
        id=str(data.get("id", "")),
        display_name=str(data.get("display_name", data.get("id", ""))),
        description=str(data.get("description", "") or ""),
        created_at=str(data.get("created_at", "") or now_iso()),
        updated_at=str(data.get("updated_at", "") or now_iso()),
    )


def target_to_dict(target: Target) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": target.id,
        "display_name": target.display_name,
        "kind": target.kind,
        "template_path": target.template_path,
        "min_confidence": target.min_confidence,
        "search_region": target.search_region,
        "click_offset": target.click_offset,
        "created_at": target.created_at,
        "updated_at": target.updated_at,
    }
    if target.locator.method != "vision_template" or target.locator.params:
        data["locator"] = {"method": target.locator.method, "params": dict(target.locator.params)}
    return data


def target_from_dict(data: dict[str, Any]) -> Target:
    locator_data = data.get("locator") or {}
    locator = LocatorStrategy(
        method=str(locator_data.get("method", "vision_template")),
        params=dict(locator_data.get("params") or {}),
    )
    return Target(
        id=str(data.get("id", "")),
        display_name=str(data.get("display_name", data.get("id", ""))),
        kind=str(data.get("kind", "other") or "other"),
        template_path=str(data.get("template_path", "") or ""),
        min_confidence=float(data.get("min_confidence", 0.88)),
        search_region=str(data.get("search_region", "active_screen") or "active_screen"),
        click_offset=str(data.get("click_offset", "center") or "center"),
        locator=locator,
        created_at=str(data.get("created_at", "") or now_iso()),
        updated_at=str(data.get("updated_at", "") or now_iso()),
    )


_STEP_FIELDS = {
    "id",
    "action",
    "target_id",
    "target_ids",
    "text",
    "keys",
    "seconds",
    "timeout_seconds",
    "message",
    "continue_on_failure",
    "params",
}


def step_to_dict(step: WorkflowStep) -> dict[str, Any]:
    data: dict[str, Any] = {"id": step.id, "action": step.action}
    if step.target_id is not None:
        data["target_id"] = step.target_id
    if step.target_ids:
        data["target_ids"] = list(step.target_ids)
    if step.text is not None:
        data["text"] = step.text
    if step.keys:
        data["keys"] = list(step.keys)
    if step.seconds is not None:
        data["seconds"] = step.seconds
    if step.timeout_seconds != 10.0:
        data["timeout_seconds"] = step.timeout_seconds
    if step.message is not None:
        data["message"] = step.message
    if step.continue_on_failure:
        data["continue_on_failure"] = True
    if step.params:
        data["params"] = dict(step.params)
    return data


def step_from_dict(data: dict[str, Any]) -> WorkflowStep:
    # Unknown keys are kept in params so hand-edited YAML never loses data.
    extras = {k: v for k, v in data.items() if k not in _STEP_FIELDS}
    params = dict(data.get("params") or {})
    params.update(extras)
    seconds = data.get("seconds")
    return WorkflowStep(
        id=str(data.get("id", "")),
        action=str(data.get("action", "")),
        target_id=data.get("target_id"),
        target_ids=[str(t) for t in (data.get("target_ids") or [])],
        text=data.get("text"),
        keys=[str(k) for k in (data.get("keys") or [])],
        seconds=float(seconds) if seconds is not None else None,
        timeout_seconds=float(data.get("timeout_seconds", 10.0)),
        message=data.get("message"),
        continue_on_failure=bool(data.get("continue_on_failure", False)),
        params=params,
    )


def workflow_to_dict(workflow: Workflow) -> dict[str, Any]:
    data: dict[str, Any] = {"id": workflow.id, "display_name": workflow.display_name}
    if workflow.description:
        data["description"] = workflow.description
    if workflow.countdown_seconds is not None:
        data["countdown_seconds"] = workflow.countdown_seconds
    if not workflow.stop_on_failure:
        data["stop_on_failure"] = False
    data["steps"] = [step_to_dict(step) for step in workflow.steps]
    return data


def workflow_from_dict(data: dict[str, Any]) -> Workflow:
    countdown = data.get("countdown_seconds")
    return Workflow(
        id=str(data.get("id", "")),
        display_name=str(data.get("display_name", data.get("id", ""))),
        description=str(data.get("description", "") or ""),
        countdown_seconds=float(countdown) if countdown is not None else None,
        stop_on_failure=bool(data.get("stop_on_failure", True)),
        steps=[step_from_dict(s) for s in (data.get("steps") or [])],
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_step(step: WorkflowStep) -> list[str]:
    """Return a list of human-readable problems with this step (empty = valid)."""
    errors: list[str] = []
    label = f"step '{step.id}'" if step.id else "step (missing id)"
    if not step.id:
        errors.append("a step is missing an 'id'")
    if step.action not in VALID_ACTIONS:
        errors.append(
            f"{label}: unknown action {step.action!r}. "
            f"Valid actions: {', '.join(sorted(VALID_ACTIONS))}"
        )
        return errors
    if step.action in ("find_target", "wait_for_target", "click_target") and not step.target_id:
        errors.append(f"{label}: action '{step.action}' requires 'target_id'")
    if step.action == "wait_for_any_target" and not step.target_ids:
        errors.append(f"{label}: action 'wait_for_any_target' requires a 'target_ids' list")
    if step.action == "paste_text" and step.text is None:
        errors.append(f"{label}: action 'paste_text' requires 'text'")
    if step.action == "hotkey" and not step.keys:
        errors.append(f"{label}: action 'hotkey' requires a 'keys' list, e.g. [command, v]")
    if step.action == "wait_seconds" and (step.seconds is None or step.seconds <= 0):
        errors.append(f"{label}: action 'wait_seconds' requires 'seconds' > 0")
    if step.timeout_seconds <= 0:
        errors.append(f"{label}: 'timeout_seconds' must be > 0")
    return errors


def referenced_target_ids(workflow: Workflow) -> set[str]:
    refs: set[str] = set()
    for step in workflow.steps:
        if step.target_id:
            refs.add(step.target_id)
        refs.update(step.target_ids)
    return refs


def validate_workflow(workflow: Workflow, known_target_ids: set[str] | None = None) -> list[str]:
    """Validate a workflow; optionally check target references against a profile."""
    errors: list[str] = []
    if not workflow.id:
        errors.append("workflow is missing an 'id'")
    if not workflow.steps:
        errors.append("workflow has no steps")
    seen: set[str] = set()
    for step in workflow.steps:
        if step.id in seen:
            errors.append(f"duplicate step id '{step.id}'")
        seen.add(step.id)
        errors.extend(validate_step(step))
    if known_target_ids is not None:
        missing = referenced_target_ids(workflow) - known_target_ids
        for target_id in sorted(missing):
            errors.append(
                f"workflow references unknown target '{target_id}' "
                "(capture it first, or fix the target_id)"
            )
    return errors
