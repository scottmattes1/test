"""macOS permission detection and user-facing help text.

We never try to bypass macOS security; we just detect the common failure
modes (screenshots failing or coming back blank, input events silently
ignored) and tell the user which System Settings switch to flip.
"""

from __future__ import annotations

import sys


def is_macos() -> bool:
    return sys.platform == "darwin"


MACOS_PERMISSIONS_HINT = (
    "Local RPA Studio needs macOS permissions to see the screen and control "
    "the mouse/keyboard. Open System Settings → Privacy & Security and enable "
    "your terminal or Python app under:\n"
    "  • Screen Recording — required for screenshots and target matching\n"
    "  • Accessibility — required for mouse clicks and keyboard input\n"
    "  • Input Monitoring — only if keyboard actions still fail\n"
    "After changing a permission, fully quit and restart the app."
)

GENERIC_CAPTURE_HINT = (
    "Screen capture failed. Make sure you are running this app in a desktop "
    "session (not over SSH / headless)."
)


def screen_capture_hint() -> str:
    return MACOS_PERMISSIONS_HINT if is_macos() else GENERIC_CAPTURE_HINT


def check_screen_capture() -> tuple[bool, str]:
    """Try one screenshot and report (ok, user-facing message)."""
    from .screen import RealScreen, ScreenCaptureError

    try:
        shot = RealScreen().screenshot()
    except ScreenCaptureError as exc:
        return False, str(exc)
    if is_macos() and float(shot.std()) < 1.0:
        # A uniform (usually black) screenshot on macOS almost always means
        # Screen Recording permission is missing.
        return False, (
            "The screenshot came back blank, which usually means Screen "
            "Recording permission is missing.\n\n" + MACOS_PERMISSIONS_HINT
        )
    return True, f"Screen capture OK ({shot.shape[1]}x{shot.shape[0]} px)."
