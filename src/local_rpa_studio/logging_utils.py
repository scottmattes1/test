"""JSONL run logging.

Every run gets its own folder (``runs/<timestamp>/``) containing ``run.jsonl``
plus any artifacts (failure screenshots, debug images). Each JSONL line is one
event: run_started, countdown, step_started, step_finished, run_finished, ...
JSONL was chosen over SQLite because runs are append-only, human-inspectable,
and trivially greppable; there is no cross-run querying in the MVP.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from .models import now_iso

RUN_LOG_FILENAME = "run.jsonl"


class RunLogger:
    """Appends one JSON object per line to run.jsonl; optionally echoes events."""

    def __init__(
        self,
        run_dir: Path,
        workflow_id: str,
        profile_id: str,
        dry_run: bool,
        echo: Callable[[dict], None] | None = None,
    ):
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / RUN_LOG_FILENAME
        self.workflow_id = workflow_id
        self.profile_id = profile_id
        self.dry_run = dry_run
        self.echo = echo

    def log(self, event_type: str, **fields) -> dict:
        event: dict = {
            "timestamp": now_iso(),
            "type": event_type,
            "workflow_id": self.workflow_id,
            "profile_id": self.profile_id,
            "dry_run": self.dry_run,
        }
        event.update({k: v for k, v in fields.items() if v is not None})
        # Open in append mode per event so a crash never loses earlier lines.
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        if self.echo is not None:
            self.echo(event)
        return event


def format_event(event: dict) -> str:
    """Render an event as a one-line human-readable string for the UI log panel."""
    timestamp = str(event.get("timestamp", ""))
    clock = timestamp[11:19] if len(timestamp) >= 19 else timestamp
    event_type = event.get("type", "event")
    message = event.get("message", "")

    if event_type == "step_finished":
        status = str(event.get("status", "")).upper()
        parts = [f"[{clock}] {status:9} {event.get('step_id')} ({event.get('action')})"]
        if message:
            parts.append(str(message))
        confidence = event.get("confidence")
        if confidence is not None:
            parts.append(f"[confidence {confidence:.3f}]")
        if event.get("error"):
            parts.append(f"error: {event['error']}")
        return " ".join(parts)
    if event_type == "step_started":
        return f"[{clock}] ▶ {event.get('step_id')} ({event.get('action')})"
    if event_type in ("run_started", "run_finished", "countdown", "info", "run_failed"):
        return f"[{clock}] {message}" if message else f"[{clock}] {event_type}"
    return f"[{clock}] {event_type}: {message}"


def read_run_log(run_dir: Path) -> list[dict]:
    """Load all events from a run folder (for tests/inspection)."""
    path = Path(run_dir) / RUN_LOG_FILENAME
    events = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events
