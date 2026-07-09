"""Shared test fixtures: synthetic images, fake screen, spy input controller."""

from __future__ import annotations

import numpy as np
import pytest

from local_rpa_studio.models import Target
from local_rpa_studio.storage import ProfileStore, Workspace


def make_background(width: int = 640, height: int = 400, seed: int = 7) -> np.ndarray:
    """A textured (noisy) background so false matches score low."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 60, size=(height, width, 3), dtype=np.uint8)


def make_button(width: int = 90, height: int = 36) -> np.ndarray:
    """A distinctive synthetic 'button' patch with borders and texture."""
    patch = np.zeros((height, width, 3), dtype=np.uint8)
    patch[:, :] = (60, 120, 200)
    patch[:3, :] = 255
    patch[-3:, :] = 255
    patch[:, :3] = 255
    patch[:, -3:] = 255
    patch[10 : height - 10, 10:30] = (230, 240, 250)
    for i in range(min(height, width)):
        patch[i, i] = (0, 0, 0)
    return patch


def stamp(canvas: np.ndarray, patch: np.ndarray, x: int, y: int) -> None:
    height, width = patch.shape[:2]
    canvas[y : y + height, x : x + width] = patch


class FakeScreen:
    """ScreenSource backed by a fixed synthetic image (scale factor 1.0)."""

    def __init__(self, image: np.ndarray):
        self.image = image

    def screenshot(self) -> np.ndarray:
        return self.image.copy()

    def logical_size(self) -> tuple[int, int]:
        return (self.image.shape[1], self.image.shape[0])


class SpyController:
    """Records every input call; used to prove dry runs never touch input."""

    def __init__(self):
        self.calls: list[tuple] = []

    def move_to(self, x: int, y: int) -> None:
        self.calls.append(("move_to", x, y))

    def click(self, x: int, y: int) -> None:
        self.calls.append(("click", x, y))

    def hotkey(self, *keys: str) -> None:
        self.calls.append(("hotkey", *keys))

    def paste_text(self, text: str) -> None:
        self.calls.append(("paste_text", text))

    def copy_selection(self) -> str:
        self.calls.append(("copy_selection",))
        return "spy-clipboard"


@pytest.fixture
def workspace(tmp_path) -> Workspace:
    ws = Workspace(tmp_path / "workspace")
    ws.ensure()
    return ws


@pytest.fixture
def profile_store(workspace) -> ProfileStore:
    return workspace.create_profile("Test App", profile_id="test_app")


@pytest.fixture
def button_patch() -> np.ndarray:
    return make_button()


@pytest.fixture
def button_target(profile_store, button_patch) -> Target:
    target = Target(id="ok_button", display_name="OK Button", min_confidence=0.85)
    profile_store.save_target(target, image_bgr=button_patch)
    return target
