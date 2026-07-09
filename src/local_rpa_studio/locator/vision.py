"""OpenCV template-matching locator.

Uses normalized cross-correlation (``cv2.TM_CCOEFF_NORMED``) on grayscale
images. Simple multi-scale matching is included: the template is tried at
several scale factors and the best-scoring scale wins. This covers the common
Retina/high-DPI mismatch (template captured at 2x physical pixels, matched
against a 1x screen, or vice versa).

TODO(robust-scaling): for production-grade robustness across arbitrary DPI /
zoom levels, replace the fixed scale ladder with an image-pyramid search or a
feature-based matcher (ORB/SIFT + homography), and consider edge-based
matching to reduce sensitivity to theme/colour changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .base import Locator, MatchResult, register_locator

# 1.0 first so exact-scale matches are found immediately; 0.5 and 2.0 cover
# Retina (2x) templates on non-Retina screens and vice versa.
DEFAULT_SCALES: tuple[float, ...] = (1.0, 0.75, 1.25, 1.5, 2.0, 0.5)

_MIN_TEMPLATE_SIDE = 4


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


@register_locator
class VisionTemplateLocator(Locator):
    method_name = "vision_template"

    def __init__(self, scales: tuple[float, ...] = DEFAULT_SCALES):
        self.scales: tuple[float, ...] = tuple(scales) or (1.0,)

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> VisionTemplateLocator:
        scales = params.get("scales")
        if scales:
            return cls(scales=tuple(float(s) for s in scales))
        return cls()

    def find_in_image(
        self, image: np.ndarray, template: np.ndarray, min_confidence: float
    ) -> MatchResult:
        gray_image = _to_gray(image)
        gray_template = _to_gray(template)
        image_h, image_w = gray_image.shape[:2]

        best: tuple[float, tuple[int, int], int, int, float] | None = None
        for scale in self.scales:
            if scale == 1.0:
                scaled = gray_template
            else:
                scaled = cv2.resize(
                    gray_template,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
                )
            tpl_h, tpl_w = scaled.shape[:2]
            if (
                tpl_h < _MIN_TEMPLATE_SIDE
                or tpl_w < _MIN_TEMPLATE_SIDE
                or tpl_h > image_h
                or tpl_w > image_w
            ):
                continue
            result = cv2.matchTemplate(gray_image, scaled, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if not np.isfinite(max_val):
                continue
            if best is None or max_val > best[0]:
                best = (float(max_val), (int(max_loc[0]), int(max_loc[1])), tpl_w, tpl_h, scale)

        if best is None:
            return MatchResult(
                found=False,
                message=(
                    "Template could not be matched at any scale (it may be larger than the screen)."
                ),
            )

        confidence, (x, y), width, height, scale = best
        center = (x + width // 2, y + height // 2)
        found = confidence >= min_confidence
        message = (
            f"best confidence {confidence:.3f} (min {min_confidence:.2f}, template scale {scale:g})"
        )
        return MatchResult(
            found=found,
            confidence=confidence,
            box=(x, y, width, height),
            center=center,
            scale=scale,
            message=message,
        )


def draw_debug(image: np.ndarray, match: MatchResult) -> np.ndarray:
    """Return a copy of the screenshot with the match box and confidence drawn."""
    annotated = image.copy()
    if annotated.ndim == 2:
        annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)
    if match.box is not None:
        x, y, width, height = match.box
        colour = (0, 200, 0) if match.found else (0, 0, 255)
        cv2.rectangle(annotated, (x, y), (x + width, y + height), colour, 2)
        label = f"{match.confidence:.3f}"
        label_y = y - 8 if y >= 20 else y + height + 20
        cv2.putText(
            annotated, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA
        )
    return annotated


def save_debug_image(path: Path, image: np.ndarray, match: MatchResult) -> None:
    cv2.imwrite(str(path), draw_debug(image, match))
