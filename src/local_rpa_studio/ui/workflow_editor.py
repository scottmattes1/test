"""Workflow editor widget: a YAML text editor with validation.

Editing the YAML directly (with a validate button and inline error feedback)
was chosen over a form/flowchart builder for the MVP — it is fully general,
keeps files as the source of truth, and preserves user comments.
"""

from __future__ import annotations

import yaml
from PySide6.QtCore import Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..models import Workflow, validate_workflow, workflow_from_dict
from ..storage import ProfileStore, StorageError

NEW_WORKFLOW_TEMPLATE = """\
id: {workflow_id}
display_name: {display_name}
description: Describe what this workflow does.
# countdown_seconds: 3      # seconds to focus the target app before the run
# stop_on_failure: true     # default: stop at the first failed step
steps:
  # Available actions: wait_seconds, find_target, wait_for_target,
  # wait_for_any_target, click_target, paste_text, copy_clipboard,
  # hotkey, screenshot, notify, stop
  - id: wait_for_button
    action: wait_for_target
    target_id: CHANGE_ME
    timeout_seconds: 10
  - id: click_button
    action: click_target
    target_id: CHANGE_ME
    timeout_seconds: 10
  - id: paste_value
    action: paste_text
    text: "Hello from Local RPA Studio"
  - id: final_wait
    action: wait_seconds
    seconds: 1
"""


class WorkflowEditor(QWidget):
    workflow_saved = Signal(str)  # workflow_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self._store: ProfileStore | None = None
        self._workflow_id: str | None = None

        self.header = QLabel("No workflow selected")
        self.editor = QPlainTextEdit()
        font = QFont("Menlo")
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.editor.setFont(font)
        self.editor.setPlaceholderText("Select or create a workflow to edit its YAML…")
        self.editor.setEnabled(False)

        self.validate_button = QPushButton("Validate")
        self.save_button = QPushButton("Save")
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)

        self.validate_button.clicked.connect(self.validate_current)
        self.save_button.clicked.connect(self.save_current)

        buttons = QHBoxLayout()
        buttons.addWidget(self.validate_button)
        buttons.addWidget(self.save_button)
        buttons.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.header)
        layout.addWidget(self.editor, 1)
        layout.addLayout(buttons)
        layout.addWidget(self.status_label)

    # ------------------------------------------------------------------

    def set_store(self, store: ProfileStore | None) -> None:
        self._store = store
        self.clear()

    def clear(self) -> None:
        self._workflow_id = None
        self.editor.clear()
        self.editor.setEnabled(False)
        self.header.setText("No workflow selected")
        self.status_label.clear()

    def load_workflow(self, workflow_id: str) -> None:
        if self._store is None:
            return
        try:
            text = self._store.read_workflow_text(workflow_id)
        except StorageError as exc:
            self._set_status(str(exc), error=True)
            return
        self._workflow_id = workflow_id
        self.editor.setPlainText(text)
        self.editor.setEnabled(True)
        self.editor.document().setModified(False)
        self.header.setText(f"Workflow: {workflow_id}.yaml")
        self.status_label.clear()

    @property
    def workflow_id(self) -> str | None:
        return self._workflow_id

    def is_modified(self) -> bool:
        return self._workflow_id is not None and self.editor.document().isModified()

    # ------------------------------------------------------------------

    def parse_current(self) -> tuple[Workflow | None, list[str]]:
        """Parse the editor text; returns (workflow, problems)."""
        try:
            data = yaml.safe_load(self.editor.toPlainText())
        except yaml.YAMLError as exc:
            return None, [f"YAML syntax error: {exc}"]
        if not isinstance(data, dict):
            return None, ["Workflow YAML must be a mapping with id/display_name/steps."]
        workflow = workflow_from_dict(data)
        known_ids = set(self._store.list_target_ids()) if self._store else None
        return workflow, validate_workflow(workflow, known_target_ids=known_ids)

    def validate_current(self) -> None:
        if self._workflow_id is None:
            return
        workflow, problems = self.parse_current()
        if workflow is None:
            self._set_status(problems[0], error=True)
        elif problems:
            self._set_status("Problems:\n• " + "\n• ".join(problems), error=True)
        else:
            self._set_status(f"Valid ✓ ({len(workflow.steps)} steps)", error=False)

    def save_current(self) -> None:
        if self._store is None or self._workflow_id is None:
            return
        workflow, problems = self.parse_current()
        if workflow is None:
            # Unparseable YAML is never written to disk.
            self._set_status("Not saved — " + problems[0], error=True)
            return
        self._store.write_workflow_text(self._workflow_id, self.editor.toPlainText())
        self.editor.document().setModified(False)
        if problems:
            self._set_status("Saved, but with problems:\n• " + "\n• ".join(problems), error=True)
        else:
            self._set_status("Saved ✓", error=False)
        self.workflow_saved.emit(self._workflow_id)

    def _set_status(self, text: str, error: bool) -> None:
        self.status_label.setStyleSheet("color: #c62828;" if error else "color: #2e7d32;")
        self.status_label.setText(text)
