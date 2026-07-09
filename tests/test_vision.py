"""Vision locator tests using synthetic images."""

import cv2

from local_rpa_studio.locator import MatchResult, VisionTemplateLocator, build_locator
from local_rpa_studio.locator.vision import save_debug_image
from local_rpa_studio.models import LocatorStrategy

from .conftest import make_background, make_button, stamp


def test_exact_match_found_at_expected_center():
    canvas = make_background()
    button = make_button()
    stamp(canvas, button, x=200, y=120)

    locator = VisionTemplateLocator(scales=(1.0,))
    match = locator.find_in_image(canvas, button, min_confidence=0.9)

    assert match.found
    assert match.confidence > 0.99
    expected = (200 + button.shape[1] // 2, 120 + button.shape[0] // 2)
    assert abs(match.center[0] - expected[0]) <= 2
    assert abs(match.center[1] - expected[1]) <= 2
    assert match.scale == 1.0


def test_absent_template_is_not_found_but_reports_confidence():
    canvas = make_background(seed=1)
    button = make_button()

    match = VisionTemplateLocator(scales=(1.0,)).find_in_image(canvas, button, 0.8)

    assert not match.found
    assert match.confidence < 0.8
    assert "best confidence" in match.message


def test_never_matches_below_min_confidence():
    """The found flag (which gates clicking) respects min_confidence exactly."""
    canvas = make_background()
    button = make_button()
    stamp(canvas, button, x=50, y=50)
    match = VisionTemplateLocator(scales=(1.0,)).find_in_image(canvas, button, 1.01)
    assert not match.found  # perfect match still below an impossible threshold


def test_multi_scale_finds_resized_target():
    canvas = make_background(width=800, height=500)
    button = make_button()
    scaled_button = cv2.resize(button, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    stamp(canvas, scaled_button, x=300, y=200)

    single = VisionTemplateLocator(scales=(1.0,)).find_in_image(canvas, button, 0.75)
    multi = VisionTemplateLocator(scales=(1.0, 1.25, 1.5, 2.0)).find_in_image(canvas, button, 0.75)

    assert not single.found
    assert multi.found
    assert multi.scale == 1.5
    expected = (300 + scaled_button.shape[1] // 2, 200 + scaled_button.shape[0] // 2)
    assert abs(multi.center[0] - expected[0]) <= 4
    assert abs(multi.center[1] - expected[1]) <= 4


def test_template_larger_than_image_at_all_scales():
    canvas = make_background(width=50, height=40)
    button = make_button(width=90, height=36)
    match = VisionTemplateLocator(scales=(1.0, 2.0)).find_in_image(canvas, button, 0.8)
    assert not match.found
    assert match.box is None


def test_debug_image_saved_with_box(tmp_path):
    canvas = make_background()
    button = make_button()
    stamp(canvas, button, x=100, y=100)
    match = VisionTemplateLocator(scales=(1.0,)).find_in_image(canvas, button, 0.9)

    path = tmp_path / "debug.png"
    save_debug_image(path, canvas, match)
    assert path.is_file()
    annotated = cv2.imread(str(path))
    assert annotated is not None
    assert (annotated != canvas).any()  # the box/label changed pixels


def test_build_locator_from_strategy():
    locator = build_locator(LocatorStrategy(method="vision_template", params={"scales": [1.0]}))
    assert isinstance(locator, VisionTemplateLocator)
    assert locator.scales == (1.0,)


def test_build_locator_unknown_method():
    import pytest

    from local_rpa_studio.locator import LocatorError

    with pytest.raises(LocatorError, match="ocr"):
        build_locator(LocatorStrategy(method="ocr"))


def test_match_result_defaults():
    match = MatchResult(found=False)
    assert match.confidence == 0.0
    assert match.center is None
