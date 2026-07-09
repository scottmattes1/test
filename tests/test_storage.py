"""Workspace/profile persistence tests, plus a sanity check of the bundled example."""

from pathlib import Path

import pytest

from local_rpa_studio.models import Target, Workflow, WorkflowStep, validate_workflow
from local_rpa_studio.storage import ProfileStore, StorageError

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def test_create_and_list_profiles(workspace):
    store = workspace.create_profile("My App")
    assert store.profile_id == "my_app"
    assert workspace.list_profile_ids() == ["my_app"]
    profile = workspace.profile_store("my_app").load_profile()
    assert profile.display_name == "My App"
    with pytest.raises(StorageError):
        workspace.create_profile("My App")  # duplicate id
    with pytest.raises(StorageError):
        workspace.profile_store("nope")


def test_target_round_trip_with_template(profile_store, button_patch):
    target = Target(id="ok_button", display_name="OK", min_confidence=0.8)
    profile_store.save_target(target, image_bgr=button_patch)

    assert profile_store.list_target_ids() == ["ok_button"]
    loaded = profile_store.load_target("ok_button")
    assert loaded.display_name == "OK"
    assert loaded.min_confidence == 0.8
    assert loaded.template_path == "targets/ok_button.png"

    template = profile_store.load_template(loaded)
    assert template.shape == button_patch.shape

    profile_store.delete_target("ok_button")
    assert profile_store.list_target_ids() == []
    assert not profile_store.template_abspath(loaded).exists()


def test_load_template_missing_file(profile_store):
    target = Target(id="ghost", display_name="Ghost")
    profile_store.save_target(target)  # metadata only, no image
    with pytest.raises(StorageError, match="missing"):
        profile_store.load_template(target)


def test_workflow_round_trip(profile_store):
    workflow = Workflow(
        id="wf",
        display_name="WF",
        steps=[
            WorkflowStep(id="s1", action="wait_for_target", target_id="btn", timeout_seconds=5),
            WorkflowStep(id="s2", action="paste_text", text="hi"),
        ],
    )
    profile_store.save_workflow(workflow)
    loaded = profile_store.load_workflow("wf")
    assert loaded == workflow


def test_workflow_text_preserves_comments(profile_store):
    text = "id: wf\ndisplay_name: WF\n# a helpful comment\nsteps:\n  - id: s\n    action: stop\n"
    profile_store.write_workflow_text("wf", text)
    assert "# a helpful comment" in profile_store.read_workflow_text("wf")


def test_new_run_dirs_are_unique(profile_store):
    first = profile_store.new_run_dir()
    second = profile_store.new_run_dir()
    assert first != second
    assert first.is_dir() and second.is_dir()


def test_bundled_example_profile_is_valid():
    store = ProfileStore(EXAMPLES_DIR / "sample_profile")
    profile = store.load_profile()
    assert profile.id == "sample_profile"

    target = store.load_target("submit_button")
    assert store.template_abspath(target).is_file()
    assert store.load_template(target).size > 0

    workflow = store.load_workflow("demo_submit_form")
    errors = validate_workflow(workflow, known_target_ids=set(store.list_target_ids()))
    assert errors == []
