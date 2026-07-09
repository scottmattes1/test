"""Qt application bootstrap."""

from __future__ import annotations

import sys
from pathlib import Path

from .storage import Workspace


def run_app(workspace_root: Path | str | None = None) -> int:
    from PySide6.QtWidgets import QApplication

    from .ui.main_window import MainWindow

    workspace = Workspace(workspace_root)
    workspace.ensure()

    app = QApplication(sys.argv[:1])
    app.setApplicationName("Local RPA Studio")
    app.setOrganizationName("LocalRPAStudio")

    window = MainWindow(workspace)
    window.resize(1280, 820)
    window.show()
    return app.exec()
