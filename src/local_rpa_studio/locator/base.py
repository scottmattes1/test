"""Locator abstractions.

A *locator* finds a target inside a screenshot and reports a match. The MVP
ships one implementation (OpenCV template matching in ``vision.py``); future
methods — macOS Accessibility API, OCR/text, Playwright/DOM — implement the
same ``Locator`` interface and register themselves in ``LOCATOR_REGISTRY`` so
targets can opt in via their ``locator.method`` field without executor
changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


class LocatorError(RuntimeError):
    """Raised when a locator cannot be built or used."""


@dataclass
class MatchResult:
    """Best match for a target inside a screenshot (pixel coordinates)."""

    found: bool
    confidence: float = 0.0
    box: tuple[int, int, int, int] | None = None  # x, y, w, h
    center: tuple[int, int] | None = None
    scale: float = 1.0  # template scale factor that produced the best match
    message: str = ""


class Locator(ABC):
    """Finds a template/target inside an image."""

    method_name: str = "base"

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> Locator:
        """Build a locator from a LocatorStrategy's params dict."""
        return cls()

    @abstractmethod
    def find_in_image(
        self, image: np.ndarray, template: np.ndarray, min_confidence: float
    ) -> MatchResult:
        """Return the best match; ``found`` is True only if confidence >= min."""


LOCATOR_REGISTRY: dict[str, type[Locator]] = {}


def register_locator(cls: type[Locator]) -> type[Locator]:
    LOCATOR_REGISTRY[cls.method_name] = cls
    return cls
