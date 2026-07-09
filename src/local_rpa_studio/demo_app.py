"""Tiny demo target app to practise automation against.

Run with ``rpa-demo-app`` (or ``python -m local_rpa_studio.demo_app``), then use
Local RPA Studio to capture its Submit button and drive it with a workflow.
"""

from __future__ import annotations

import sys


def main() -> int:
    from PySide6.QtWidgets import (
        QApplication,
        QLabel,
        QLineEdit,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )

    app = QApplication(sys.argv[:1])
    window = QWidget()
    window.setWindowTitle("RPA Demo Target")

    field = QLineEdit()
    field.setPlaceholderText("Click here, then let the workflow paste text…")
    status = QLabel("Waiting…")
    status.setStyleSheet("color: #555;")
    counter = {"clicks": 0}

    submit = QPushButton("Submit")
    submit.setObjectName("submit")
    submit.setStyleSheet(
        "QPushButton#submit { background-color: #2e7d32; color: white; font-weight: bold;"
        " font-size: 15px; padding: 10px 28px; border-radius: 6px; }"
        "QPushButton#submit:hover { background-color: #1b5e20; }"
    )

    reset = QPushButton("Reset")

    def on_submit() -> None:
        counter["clicks"] += 1
        status.setText(f"Submitted #{counter['clicks']}: “{field.text()}”")
        status.setStyleSheet("color: #2e7d32; font-weight: bold;")

    def on_reset() -> None:
        field.clear()
        status.setText("Waiting…")
        status.setStyleSheet("color: #555;")

    submit.clicked.connect(on_submit)
    reset.clicked.connect(on_reset)

    layout = QVBoxLayout(window)
    layout.addWidget(QLabel("Demo form — capture the Submit button as a target:"))
    layout.addWidget(field)
    layout.addWidget(submit)
    layout.addWidget(reset)
    layout.addWidget(status)

    window.resize(420, 220)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
