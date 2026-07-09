"""Main application window: profiles, targets, workflows, runs, logs."""

from __future__ import annotations

import time

from PySide6.QtCore import Qt, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..executor.runner import WorkflowRunner
from ..logging_utils import format_event
from ..models import RunResult, Workflow, slugify
from ..permissions import check_screen_capture
from ..screen import RealScreen, ScreenCaptureError
from ..storage import ProfileStore, StorageError, Workspace
from .capture_overlay import RegionSelector, TargetDetailsDialog
from .workflow_editor import NEW_WORKFLOW_TEMPLATE, WorkflowEditor


class RunThread(QThread):
    """Executes a workflow off the UI thread, relaying events via signals."""

    event_received = Signal(dict)
    run_completed = Signal(object)  # RunResult

    def __init__(self, store: ProfileStore, workflow: Workflow, dry_run: bool, save_debug: bool):
        super().__init__()
        # save_debug=None lets the runner pick its default (on for dry runs).
        self.runner = WorkflowRunner(
            store,
            workflow,
            dry_run=dry_run,
            save_debug=True if save_debug else None,
            on_event=self.event_received.emit,
        )

    def run(self) -> None:
        try:
            result = self.runner.run()
        except Exception as exc:  # defensive: never let the UI thread hang
            self.event_received.emit(
                {"type": "run_failed", "timestamp": "", "message": f"Runner crashed: {exc!r}"}
            )
            result = None
        self.run_completed.emit(result)


class MainWindow(QMainWindow):
    def __init__(self, workspace: Workspace):
        super().__init__()
        self.workspace = workspace
        self.store: ProfileStore | None = None
        self.run_thread: RunThread | None = None

        self.setWindowTitle("Local RPA Studio")
        self._build_ui()
        self._refresh_profiles()

    # -- UI construction ------------------------------------------------

    def _build_ui(self) -> None:
        # Top bar: profile selection
        self.profile_combo = QComboBox()
        self.profile_combo.currentIndexChanged.connect(self._on_profile_changed)
        new_profile_button = QPushButton("New Profile…")
        new_profile_button.clicked.connect(self._new_profile)
        open_folder_button = QPushButton("Open Profile Folder")
        open_folder_button.clicked.connect(self._open_profile_folder)
        permissions_button = QPushButton("Check Permissions")
        permissions_button.clicked.connect(self._check_permissions)

        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("Profile:"))
        top_bar.addWidget(self.profile_combo, 1)
        top_bar.addWidget(new_profile_button)
        top_bar.addWidget(open_folder_button)
        top_bar.addWidget(permissions_button)

        # Targets panel
        self.target_list = QListWidget()
        self.target_list.itemDoubleClicked.connect(self._edit_target)
        capture_button = QPushButton("Capture Target…")
        capture_button.clicked.connect(self._capture_target)
        delete_target_button = QPushButton("Delete")
        delete_target_button.clicked.connect(self._delete_target)
        target_buttons = QHBoxLayout()
        target_buttons.addWidget(capture_button)
        target_buttons.addWidget(delete_target_button)
        targets_box = QGroupBox("Targets (double-click to edit)")
        targets_layout = QVBoxLayout(targets_box)
        targets_layout.addWidget(self.target_list, 1)
        targets_layout.addLayout(target_buttons)

        # Workflows panel
        self.workflow_list = QListWidget()
        self.workflow_list.currentItemChanged.connect(self._on_workflow_selected)
        new_workflow_button = QPushButton("New Workflow…")
        new_workflow_button.clicked.connect(self._new_workflow)
        delete_workflow_button = QPushButton("Delete")
        delete_workflow_button.clicked.connect(self._delete_workflow)
        workflow_buttons = QHBoxLayout()
        workflow_buttons.addWidget(new_workflow_button)
        workflow_buttons.addWidget(delete_workflow_button)
        workflows_box = QGroupBox("Workflows")
        workflows_layout = QVBoxLayout(workflows_box)
        workflows_layout.addWidget(self.workflow_list, 1)
        workflows_layout.addLayout(workflow_buttons)

        # Run controls
        self.run_button = QPushButton("▶ Run")
        self.dry_run_button = QPushButton("Dry Run")
        self.cancel_button = QPushButton("Cancel Run")
        self.cancel_button.setEnabled(False)
        self.debug_checkbox = QCheckBox("Save debug images")
        self.run_button.clicked.connect(lambda: self._start_run(dry_run=False))
        self.dry_run_button.clicked.connect(lambda: self._start_run(dry_run=True))
        self.cancel_button.clicked.connect(self._cancel_run)
        open_runs_button = QPushButton("Open Runs Folder")
        open_runs_button.clicked.connect(self._open_runs_folder)
        run_bar = QHBoxLayout()
        run_bar.addWidget(self.run_button)
        run_bar.addWidget(self.dry_run_button)
        run_bar.addWidget(self.cancel_button)
        run_bar.addWidget(self.debug_checkbox)
        run_bar.addStretch(1)
        run_bar.addWidget(open_runs_button)

        # Editor + log
        self.editor = WorkflowEditor()
        self.editor.workflow_saved.connect(
            lambda _wid: self._refresh_workflows(keep_selection=True)
        )
        self.log_panel = QPlainTextEdit()
        self.log_panel.setReadOnly(True)
        log_font = QFont("Menlo")
        log_font.setStyleHint(QFont.StyleHint.Monospace)
        self.log_panel.setFont(log_font)
        log_box = QGroupBox("Run log")
        log_layout = QVBoxLayout(log_box)
        log_layout.addWidget(self.log_panel)

        left_splitter = QSplitter(Qt.Orientation.Vertical)
        left_splitter.addWidget(targets_box)
        left_splitter.addWidget(workflows_box)

        editor_widget = QWidget()
        editor_layout = QVBoxLayout(editor_widget)
        editor_layout.setContentsMargins(0, 0, 0, 0)
        editor_layout.addWidget(self.editor, 1)
        editor_layout.addLayout(run_bar)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.addWidget(left_splitter)
        main_splitter.addWidget(editor_widget)
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 2)

        vertical_splitter = QSplitter(Qt.Orientation.Vertical)
        vertical_splitter.addWidget(main_splitter)
        vertical_splitter.addWidget(log_box)
        vertical_splitter.setStretchFactor(0, 3)
        vertical_splitter.setStretchFactor(1, 1)

        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.addLayout(top_bar)
        central_layout.addWidget(vertical_splitter, 1)
        self.setCentralWidget(central)
        self.statusBar().showMessage(f"Workspace: {self.workspace.root}")

    # -- profiles ---------------------------------------------------------

    def _refresh_profiles(self, select_id: str | None = None) -> None:
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        profile_ids = self.workspace.list_profile_ids()
        for profile_id in profile_ids:
            self.profile_combo.addItem(profile_id)
        self.profile_combo.blockSignals(False)
        if not profile_ids:
            self.store = None
            self.editor.set_store(None)
            self.target_list.clear()
            self.workflow_list.clear()
            self._log_line("No profiles yet — click “New Profile…” to create one.")
            return
        index = profile_ids.index(select_id) if select_id in profile_ids else 0
        self.profile_combo.setCurrentIndex(index)
        self._on_profile_changed(index)

    def _on_profile_changed(self, _index: int) -> None:
        profile_id = self.profile_combo.currentText()
        if not profile_id:
            return
        try:
            self.store = self.workspace.profile_store(profile_id)
        except StorageError as exc:
            QMessageBox.critical(self, "Profile error", str(exc))
            return
        self.editor.set_store(self.store)
        self._refresh_targets()
        self._refresh_workflows()

    def _new_profile(self) -> None:
        name, ok = QInputDialog.getText(self, "New Profile", "Profile name (e.g. QuickBooks):")
        if not ok or not name.strip():
            return
        try:
            store = self.workspace.create_profile(name.strip())
        except StorageError as exc:
            QMessageBox.warning(self, "Could not create profile", str(exc))
            return
        self._refresh_profiles(select_id=store.profile_id)
        self._log_line(f"Created profile '{store.profile_id}' at {store.profile_dir}")

    # -- targets ----------------------------------------------------------

    def _refresh_targets(self) -> None:
        self.target_list.clear()
        if self.store is None:
            return
        try:
            targets = self.store.list_targets()
        except StorageError as exc:
            QMessageBox.warning(self, "Target error", str(exc))
            return
        for target in targets:
            item = QListWidgetItem(
                f"{target.display_name}  ({target.id} · {target.kind} · "
                f"min {target.min_confidence:.2f})"
            )
            item.setData(Qt.ItemDataRole.UserRole, target.id)
            self.target_list.addItem(item)

    def _selected_target_id(self) -> str | None:
        item = self.target_list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def _capture_target(self) -> None:
        if self.store is None:
            QMessageBox.information(self, "No profile", "Create or select a profile first.")
            return
        # Hide the studio window so it doesn't cover the app being captured.
        self.hide()
        QApplication.processEvents()
        time.sleep(0.4)  # give the window manager time to actually hide us
        try:
            screenshot = RealScreen().screenshot()
        except ScreenCaptureError as exc:
            self.show()
            QMessageBox.critical(self, "Screen capture failed", str(exc))
            return

        selector = RegionSelector(screenshot)
        accepted = selector.exec()
        self.show()
        self.raise_()
        self.activateWindow()
        if not accepted:
            return
        crop = selector.selected_image()
        if crop is None:
            QMessageBox.information(
                self, "Selection too small", "Please drag a larger rectangle (≥ 8×8 px)."
            )
            return

        dialog = TargetDetailsDialog(existing_ids=set(self.store.list_target_ids()), parent=self)
        if not dialog.exec():
            return
        target = dialog.result_target()
        try:
            self.store.save_target(target, image_bgr=crop)
        except StorageError as exc:
            QMessageBox.critical(self, "Could not save target", str(exc))
            return
        self._refresh_targets()
        self._log_line(
            f"Captured target '{target.id}' ({crop.shape[1]}x{crop.shape[0]} px) "
            f"→ {target.template_path}"
        )

    def _edit_target(self, item: QListWidgetItem) -> None:
        if self.store is None:
            return
        target_id = item.data(Qt.ItemDataRole.UserRole)
        try:
            target = self.store.load_target(target_id)
        except StorageError as exc:
            QMessageBox.warning(self, "Target error", str(exc))
            return
        dialog = TargetDetailsDialog(existing=target, parent=self)
        if dialog.exec():
            self.store.save_target(dialog.result_target())
            self._refresh_targets()

    def _delete_target(self) -> None:
        target_id = self._selected_target_id()
        if self.store is None or target_id is None:
            return
        answer = QMessageBox.question(
            self,
            "Delete target",
            f"Delete target '{target_id}' and its template image?",
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.store.delete_target(target_id)
            self._refresh_targets()

    # -- workflows ----------------------------------------------------------

    def _refresh_workflows(self, keep_selection: bool = False) -> None:
        selected = self._selected_workflow_id() if keep_selection else None
        self.workflow_list.blockSignals(True)
        self.workflow_list.clear()
        workflow_ids = self.store.list_workflow_ids() if self.store else []
        for workflow_id in workflow_ids:
            self.workflow_list.addItem(workflow_id)
        self.workflow_list.blockSignals(False)
        if selected in workflow_ids:
            self.workflow_list.setCurrentRow(workflow_ids.index(selected))

    def _selected_workflow_id(self) -> str | None:
        item = self.workflow_list.currentItem()
        return item.text() if item else None

    def _on_workflow_selected(self, current: QListWidgetItem | None, _previous=None) -> None:
        if current is not None:
            self.editor.load_workflow(current.text())

    def _new_workflow(self) -> None:
        if self.store is None:
            QMessageBox.information(self, "No profile", "Create or select a profile first.")
            return
        name, ok = QInputDialog.getText(self, "New Workflow", "Workflow name:")
        if not ok or not name.strip():
            return
        workflow_id = slugify(name)
        if workflow_id in self.store.list_workflow_ids():
            QMessageBox.warning(
                self, "Duplicate", f"A workflow with id '{workflow_id}' already exists."
            )
            return
        self.store.write_workflow_text(
            workflow_id,
            NEW_WORKFLOW_TEMPLATE.format(workflow_id=workflow_id, display_name=name.strip()),
        )
        self._refresh_workflows()
        rows = self.store.list_workflow_ids()
        self.workflow_list.setCurrentRow(rows.index(workflow_id))

    def _delete_workflow(self) -> None:
        workflow_id = self._selected_workflow_id()
        if self.store is None or workflow_id is None:
            return
        answer = QMessageBox.question(self, "Delete workflow", f"Delete workflow '{workflow_id}'?")
        if answer == QMessageBox.StandardButton.Yes:
            self.store.delete_workflow(workflow_id)
            self.editor.clear()
            self._refresh_workflows()

    # -- runs ---------------------------------------------------------------

    def _start_run(self, dry_run: bool) -> None:
        if self.run_thread is not None and self.run_thread.isRunning():
            QMessageBox.information(self, "Busy", "A run is already in progress.")
            return
        if self.store is None:
            QMessageBox.information(self, "No profile", "Create or select a profile first.")
            return
        workflow_id = self._selected_workflow_id() or self.editor.workflow_id
        if workflow_id is None:
            QMessageBox.information(self, "No workflow", "Select or create a workflow first.")
            return
        if self.editor.is_modified():
            answer = QMessageBox.question(
                self,
                "Unsaved changes",
                "The workflow has unsaved changes. Save before running?",
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Save:
                return
            self.editor.save_current()
            if self.editor.is_modified():  # save failed (e.g. YAML syntax error)
                return
        try:
            workflow = self.store.load_workflow(workflow_id)
        except StorageError as exc:
            QMessageBox.critical(self, "Workflow error", str(exc))
            return

        mode = "DRY RUN" if dry_run else "RUN"
        self._log_line(f"――― {mode}: {workflow.display_name} ―――")
        self.run_thread = RunThread(
            self.store, workflow, dry_run=dry_run, save_debug=self.debug_checkbox.isChecked()
        )
        self.run_thread.event_received.connect(self._on_run_event)
        self.run_thread.run_completed.connect(self._on_run_completed)
        self._set_running(True)
        self.run_thread.start()

    def _cancel_run(self) -> None:
        if self.run_thread is not None and self.run_thread.isRunning():
            self.run_thread.runner.cancel()
            self._log_line("Cancel requested — finishing current step…")

    def _set_running(self, running: bool) -> None:
        self.run_button.setEnabled(not running)
        self.dry_run_button.setEnabled(not running)
        self.cancel_button.setEnabled(running)

    def _on_run_event(self, event: dict) -> None:
        self._log_line(format_event(event))

    def _on_run_completed(self, result: RunResult | None) -> None:
        self._set_running(False)
        if result is not None:
            self._log_line(f"Artifacts: {result.run_dir}")
        self.run_thread = None

    # -- misc -----------------------------------------------------------------

    def _check_permissions(self) -> None:
        ok, message = check_screen_capture()
        if ok:
            QMessageBox.information(self, "Permissions", message)
        else:
            QMessageBox.warning(self, "Permissions", message)

    def _open_profile_folder(self) -> None:
        if self.store is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.store.profile_dir)))

    def _open_runs_folder(self) -> None:
        if self.store is not None:
            self.store.runs_dir.mkdir(parents=True, exist_ok=True)
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.store.runs_dir)))

    def _log_line(self, text: str) -> None:
        self.log_panel.appendPlainText(text)

    def closeEvent(self, event) -> None:
        if self.run_thread is not None and self.run_thread.isRunning():
            self.run_thread.runner.cancel()
            self.run_thread.wait(3000)
        super().closeEvent(event)
