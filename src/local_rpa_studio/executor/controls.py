"""Real mouse/keyboard/clipboard controller.

All ``pyautogui``/``pyperclip`` imports happen lazily inside methods so tests
(and headless environments) can import this module and substitute a spy
controller. PyAutoGUI's fail-safe stays enabled: slamming the mouse into the
top-left screen corner aborts a run.
"""

from __future__ import annotations

import sys
import time

from ..permissions import screen_capture_hint


class InputControlError(RuntimeError):
    """Mouse/keyboard/clipboard control failed; message includes hints."""


def paste_modifier() -> str:
    return "command" if sys.platform == "darwin" else "ctrl"


def _import_pyautogui():
    try:
        import pyautogui
    except Exception as exc:
        raise InputControlError(
            f"PyAutoGUI could not be initialised ({exc}).\n\n{screen_capture_hint()}"
        ) from exc
    return pyautogui


def _import_pyperclip():
    try:
        import pyperclip

        return pyperclip
    except Exception as exc:
        raise InputControlError(f"Clipboard support (pyperclip) unavailable: {exc}") from exc


class InputController:
    """Performs real input. Never instantiated with side effects; every method
    imports its backend on first use and wraps failures in InputControlError."""

    move_duration = 0.2  # seconds; visible cursor travel so the user can react

    def move_to(self, x: int, y: int) -> None:
        gui = _import_pyautogui()
        try:
            gui.moveTo(x, y, duration=self.move_duration)
        except Exception as exc:
            raise InputControlError(f"Could not move the mouse ({exc}).") from exc

    def click(self, x: int, y: int) -> None:
        gui = _import_pyautogui()
        try:
            gui.click(x, y)
        except Exception as exc:
            raise InputControlError(
                f"Could not click ({exc}). On macOS, check Accessibility permission."
            ) from exc

    def hotkey(self, *keys: str) -> None:
        gui = _import_pyautogui()
        try:
            gui.hotkey(*keys)
        except Exception as exc:
            raise InputControlError(
                f"Could not press hotkey {'+'.join(keys)} ({exc}). "
                "On macOS, check Accessibility (and Input Monitoring) permissions."
            ) from exc

    def paste_text(self, text: str) -> None:
        """Put text on the clipboard and press the platform paste shortcut."""
        pyperclip = _import_pyperclip()
        try:
            pyperclip.copy(text)
        except Exception as exc:
            raise InputControlError(f"Could not write to the clipboard ({exc}).") from exc
        time.sleep(0.05)  # let the clipboard settle before pasting
        self.hotkey(paste_modifier(), "v")

    def copy_selection(self) -> str:
        """Press the platform copy shortcut and return the clipboard contents."""
        pyperclip = _import_pyperclip()
        self.hotkey(paste_modifier(), "c")
        time.sleep(0.15)  # give the frontmost app time to update the clipboard
        try:
            return pyperclip.paste()
        except Exception as exc:
            raise InputControlError(f"Could not read the clipboard ({exc}).") from exc
