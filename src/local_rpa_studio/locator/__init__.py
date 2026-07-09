"""Locator package: pluggable strategies for finding targets on screen."""

from __future__ import annotations

from ..models import LocatorStrategy
from . import vision as _vision  # noqa: F401  (registers VisionTemplateLocator)
from .base import LOCATOR_REGISTRY, Locator, LocatorError, MatchResult
from .vision import DEFAULT_SCALES, VisionTemplateLocator, draw_debug, save_debug_image

__all__ = [
    "DEFAULT_SCALES",
    "LOCATOR_REGISTRY",
    "Locator",
    "LocatorError",
    "MatchResult",
    "VisionTemplateLocator",
    "build_locator",
    "draw_debug",
    "save_debug_image",
]


def build_locator(strategy: LocatorStrategy) -> Locator:
    """Build the locator a target asks for via its ``locator.method`` field."""
    locator_cls = LOCATOR_REGISTRY.get(strategy.method)
    if locator_cls is None:
        raise LocatorError(
            f"Unknown locator method {strategy.method!r}. "
            f"Available: {', '.join(sorted(LOCATOR_REGISTRY))}. "
            "(Accessibility, OCR and Playwright locators are on the roadmap.)"
        )
    return locator_cls.from_params(strategy.params)
