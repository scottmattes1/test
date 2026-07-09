"""Screen capture and display-scale helpers.

``pyautogui`` is imported lazily inside methods so that this module (and
everything that imports it) can be used in headless environments — tests
inject a fake ``ScreenSource`` instead.

Retina/high-DPI note: on Retina Macs ``pyautogui.screenshot()`` returns an
image in *physical* pixels while the mouse works in *logical* points. We
compute ``display_scale = screenshot_width / logical_width`` and divide match
coordinates by it before moving/clicking. Template scale mismatches (e.g. a
template captured on a Retina screen matched against a non-Retina screen) are
handled by the vision locator's multi-scale matching.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .permissions import screen_capture_hint


class ScreenCaptureError(RuntimeError):
    """Screen could not be captured; message includes permission hints."""


class ScreenSource(Protocol):
    """Anything that can produce screenshots (real screen, or a fake in tests)."""

    def screenshot(self) -> np.ndarray:
        """Return the current screen as a BGR uint8 array."""
        ...

    def logical_size(self) -> tuple[int, int]:
        """Return the (width, height) of the screen in logical points."""
        ...


def _import_pyautogui():
    try:
        import pyautogui
    except Exception as exc:  # pyautogui raises non-ImportError on headless systems
        raise ScreenCaptureError(
            f"PyAutoGUI could not be initialised ({exc}).\n\n{screen_capture_hint()}"
        ) from exc
    return pyautogui


class RealScreen:
    """Captures the primary screen via PyAutoGUI."""

    def screenshot(self) -> np.ndarray:
        pyautogui = _import_pyautogui()
        try:
            pil_image = pyautogui.screenshot()
        except Exception as exc:
            raise ScreenCaptureError(
                f"Taking a screenshot failed ({exc}).\n\n{screen_capture_hint()}"
            ) from exc
        rgb = np.asarray(pil_image.convert("RGB"))
        return rgb[:, :, ::-1].copy()  # RGB -> BGR for OpenCV

    def logical_size(self) -> tuple[int, int]:
        pyautogui = _import_pyautogui()
        size = pyautogui.size()
        return (int(size.width), int(size.height))


def display_scale(screenshot: np.ndarray, logical_size: tuple[int, int]) -> float:
    """Physical-pixels-per-logical-point ratio (2.0 on typical Retina displays)."""
    logical_width = logical_size[0]
    if logical_width <= 0:
        return 1.0
    return screenshot.shape[1] / logical_width


def to_logical_point(x: float, y: float, scale: float) -> tuple[int, int]:
    """Convert screenshot pixel coordinates to logical screen coordinates."""
    if scale <= 0:
        scale = 1.0
    return (round(x / scale), round(y / scale))
