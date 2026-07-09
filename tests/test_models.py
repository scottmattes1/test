"""Model serialisation and workflow validation tests."""

import pytest

from local_rpa_studio.models import (
    Target,
    WorkflowStep,
    slugify,
    step_from_dict,
    step_to_dict,
    target_from_dict,
    target_to_dict,
    validate_step,
    validate_workflow,
    workflow_from_dict,
)


def test_slugify():
    assert slugify("Submit Button!") == "submit_button"
    assert slugify("  QuickBooks 2026  ") == "quickbooks_2026"
    assert slugify("***") == "item"


def test_target_round_trip():
    target = Target(
        id="submit_button",
        display_name="Submit Button",
        kind="button",
        template_path="targets/submit_button.png",
        min_confidence=0.9,
        click_offset="10,-4",
    )
    data = target_to_dict(target)
    loaded = target_from_dict(data)
    assert loaded == target


def test_click_offset_parsing():
    assert Target(id="a", display_name="a").resolve_click_offset() == (0, 0)
    assert Target(id="a", display_name="a", click_offset="10, -4").resolve_click_offset() == (
        10,
        -4,
    )
    with pytest.raises(ValueError):
        Target(id="a", display_name="a", click_offset="nonsense").resolve_click_offset()


def test_step_from_dict_keeps_unknown_keys_in_params():
    step = step_from_dict(
        {"id": "s1", "action": "click_target", "target_id": "btn", "custom_key": 42}
    )
    assert step.params["custom_key"] == 42
    # and round-trips through to_dict
    assert step_to_dict(step)["params"]["custom_key"] == 42


def test_validate_step_rules():
    assert validate_step(WorkflowStep(id="s", action="does_not_exist"))
    assert validate_step(WorkflowStep(id="s", action="click_target"))  # missing target_id
    assert validate_step(WorkflowStep(id="s", action="paste_text"))  # missing text
    assert validate_step(WorkflowStep(id="s", action="hotkey"))  # missing keys
    assert validate_step(WorkflowStep(id="s", action="wait_seconds"))  # missing seconds
    assert validate_step(WorkflowStep(id="s", action="wait_seconds", seconds=-1))
    assert validate_step(WorkflowStep(id="s", action="wait_for_any_target"))  # missing ids
    assert not validate_step(WorkflowStep(id="s", action="wait_seconds", seconds=1.5))
    assert not validate_step(WorkflowStep(id="s", action="click_target", target_id="btn"))
    assert not validate_step(WorkflowStep(id="s", action="stop"))


def test_validate_workflow_full():
    workflow = workflow_from_dict(
        {
            "id": "demo",
            "display_name": "Demo",
            "steps": [
                {"id": "a", "action": "wait_for_target", "target_id": "btn"},
                {"id": "a", "action": "click_target", "target_id": "missing"},
            ],
        }
    )
    errors = validate_workflow(workflow, known_target_ids={"btn"})
    assert any("duplicate step id" in e for e in errors)
    assert any("unknown target 'missing'" in e for e in errors)

    ok = workflow_from_dict(
        {
            "id": "demo",
            "display_name": "Demo",
            "steps": [{"id": "a", "action": "click_target", "target_id": "btn"}],
        }
    )
    assert validate_workflow(ok, known_target_ids={"btn"}) == []


def test_validate_workflow_requires_steps():
    workflow = workflow_from_dict({"id": "empty", "display_name": "Empty"})
    assert any("no steps" in e for e in validate_workflow(workflow))
