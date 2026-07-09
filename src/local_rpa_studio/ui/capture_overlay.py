"""Target capture UI.

Pragmatic screenshot-window approach (more reliable than a live transparent
overlay): we grab a screenshot first, display it full-screen, and let the user
drag a rectangle on the *frozen* image. The crop is taken from the original
screenshot pixels, so Retina scaling cannot skew the saved template.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QColor, QImage, QKeyEvent, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
)

from ..models import TARGET_KINDS, Target, now_iso, slugify

_MIN_SELECTION_PX = 8


def _bgr_to_qpixmap(image_bgr: np.ndarray) -> QPixmap:
    rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
    height, width, _ = rgb.shape
    qimage = QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimage.copy())  # copy: detach from the numpy buffer


class RegionSelector(QDialog):
    """Full-screen frozen-screenshot rectangle selector. Esc cancels."""

    def __init__(self, image_bgr: np.ndarray, parent=None):
        super().__init__(parent)
        self._image = image_bgr
        self._pixmap = _bgr_to_qpixmap(image_bgr)
        self._origin: QPoint | None = None
        self._current: QPoint | None = None
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setWindowState(Qt.WindowState.WindowFullScreen)

    # -- geometry -------------------------------------------------------

    def _selection_rect(self) -> QRect | None:
        if self._origin is None or self._current is None:
            return None
        return QRect(self._origin, self._current).normalized()

    def selected_image(self) -> np.ndarray | None:
        """Map the on-screen selection back to original screenshot pixels."""
        rect = self._selection_rect()
        if rect is None or self.width() == 0 or self.height() == 0:
            return None
        image_h, image_w = self._image.shape[:2]
        sx = image_w / self.width()
        sy = image_h / self.height()
        x0 = max(0, int(rect.left() * sx))
        y0 = max(0, int(rect.top() * sy))
        x1 = min(image_w, int(rect.right() * sx))
        y1 = min(image_h, int(rect.bottom() * sy))
        if x1 - x0 < _MIN_SELECTION_PX or y1 - y0 < _MIN_SELECTION_PX:
            return None
        return self._image[y0:y1, x0:x1].copy()

    # -- events ----------------------------------------------------------

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        target_rect = self.rect()
        painter.drawPixmap(target_rect, self._pixmap)
        painter.fillRect(target_rect, QColor(0, 0, 0, 90))
        selection = self._selection_rect()
        if selection is not None and self.width() and self.height():
            # Re-draw the selected part of the screenshot without dimming.
            pw = self._pixmap.width() / self.width()
            ph = self._pixmap.height() / self.height()
            source = QRect(
                int(selection.left() * pw),
                int(selection.top() * ph),
                int(selection.width() * pw),
                int(selection.height() * ph),
            )
            painter.drawPixmap(selection, self._pixmap, source)
            pen = QPen(QColor(255, 60, 60), 2)
            painter.setPen(pen)
            painter.drawRect(selection)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(
            target_rect.adjusted(0, 24, 0, 0),
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter,
            "Drag to select the target region — Esc to cancel",
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._origin = event.position().toPoint()
            self._current = self._origin
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._origin is not None:
            self._current = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton or self._origin is None:
            return
        self._current = event.position().toPoint()
        rect = self._selection_rect()
        if rect is not None and min(rect.width(), rect.height()) >= _MIN_SELECTION_PX:
            self.accept()
        else:
            self._origin = self._current = None  # too small; let the user retry
            self.update()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.reject()
        else:
            super().keyPressEvent(event)


class TargetDetailsDialog(QDialog):
    """Prompt for target id / display name / kind / min confidence."""

    def __init__(
        self,
        existing: Target | None = None,
        existing_ids: set[str] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._existing = existing
        self._existing_ids = existing_ids or set()
        self._id_edited_manually = existing is not None
        self.setWindowTitle("Edit Target" if existing else "New Target")

        self.name_edit = QLineEdit(existing.display_name if existing else "")
        self.id_edit = QLineEdit(existing.id if existing else "")
        self.kind_combo = QComboBox()
        self.kind_combo.addItems(TARGET_KINDS)
        if existing and existing.kind in TARGET_KINDS:
            self.kind_combo.setCurrentText(existing.kind)
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.50, 1.00)
        self.confidence_spin.setSingleStep(0.01)
        self.confidence_spin.setDecimals(2)
        self.confidence_spin.setValue(existing.min_confidence if existing else 0.88)

        if existing:
            # Renaming would require moving files on disk; out of MVP scope.
            self.id_edit.setEnabled(False)
        else:
            self.name_edit.textChanged.connect(self._sync_id_from_name)
            self.id_edit.textEdited.connect(self._mark_id_edited)

        form = QFormLayout()
        form.addRow("Display name:", self.name_edit)
        form.addRow("Target id:", self.id_edit)
        form.addRow("Kind:", self.kind_combo)
        form.addRow("Min confidence:", self.confidence_spin)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def _sync_id_from_name(self, text: str) -> None:
        if not self._id_edited_manually:
            self.id_edit.setText(slugify(text))

    def _mark_id_edited(self) -> None:
        self._id_edited_manually = True

    def _validate_and_accept(self) -> None:
        target_id = slugify(self.id_edit.text())
        if not self.name_edit.text().strip():
            QMessageBox.warning(self, "Missing name", "Please enter a display name.")
            return
        if not target_id:
            QMessageBox.warning(self, "Missing id", "Please enter a target id.")
            return
        if self._existing is None and target_id in self._existing_ids:
            QMessageBox.warning(
                self, "Duplicate id", f"A target with id '{target_id}' already exists."
            )
            return
        self.id_edit.setText(target_id)
        self.accept()

    def result_target(self) -> Target:
        if self._existing is not None:
            self._existing.display_name = self.name_edit.text().strip()
            self._existing.kind = self.kind_combo.currentText()
            self._existing.min_confidence = round(self.confidence_spin.value(), 2)
            self._existing.updated_at = now_iso()
            return self._existing
        return Target(
            id=self.id_edit.text(),
            display_name=self.name_edit.text().strip(),
            kind=self.kind_combo.currentText(),
            min_confidence=round(self.confidence_spin.value(), 2),
        )
