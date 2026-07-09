"""Workflow runner tests: dry-run safety, normal execution, failures, cancellation."""

import threading

from local_rpa_studio.executor.runner import WorkflowRunner
from local_rpa_studio.logging_utils import read_run_log
from local_rpa_studio.models import (
    STATUS_CANCELLED,
    STATUS_FAILURE,
    STATUS_SKIPPED,
    STATUS_SUCCESS,
    Workflow,
    WorkflowStep,
)

from .conftest import FakeScreen, SpyController, make_background, stamp

BUTTON_POS = (200, 120)


def make_workflow(steps) -> Workflow:
    return Workflow(id="wf", display_name="Test Workflow", steps=steps)


def make_runner(profile_store, workflow, screen, *, dry_run, controller=None, **kwargs):
    return WorkflowRunner(
        profile_store,
        workflow,
        dry_run=dry_run,
        screen=screen,
        controller=controller or SpyController(),
        countdown_seconds=0,
        poll_interval=0.05,
        pre_click_pause=0.0,
        **kwargs,
    )


def screen_with_button(button_patch) -> FakeScreen:
    canvas = make_background()
    stamp(canvas, button_patch, x=BUTTON_POS[0], y=BUTTON_POS[1])
    return FakeScreen(canvas)


def full_workflow() -> Workflow:
    return make_workflow(
        [
            WorkflowStep(
                id="wait", action="wait_for_target", target_id="ok_button", timeout_seconds=2
            ),
            WorkflowStep(
                id="click", action="click_target", target_id="ok_button", timeout_seconds=2
            ),
            WorkflowStep(id="paste", action="paste_text", text="hello"),
            WorkflowStep(id="keys", action="hotkey", keys=["ctrl", "s"]),
            WorkflowStep(id="pause", action="wait_seconds", seconds=0.05),
            WorkflowStep(id="note", action="notify", message="done"),
        ]
    )


def test_dry_run_performs_no_input_actions(profile_store, button_target, button_patch):
    spy = SpyController()
    runner = make_runner(
        profile_store,
        full_workflow(),
        screen_with_button(button_patch),
        dry_run=True,
        controller=spy,
    )
    result = runner.run()

    assert result.status == STATUS_SUCCESS
    assert all(r.status == STATUS_SUCCESS for r in result.step_results)
    assert spy.calls == []  # the core dry-run guarantee

    click_result = next(r for r in result.step_results if r.step_id == "click")
    assert "would click" in click_result.message
    assert click_result.confidence is not None and click_result.confidence > 0.9
    assert click_result.location is not None

    events = read_run_log(result.run_dir)
    finished = [e for e in events if e["type"] == "step_finished"]
    assert len(finished) == 6
    assert all(e["status"] == STATUS_SUCCESS for e in finished)
    assert all(e["dry_run"] is True for e in events)


def test_dry_run_saves_debug_images(profile_store, button_target, button_patch):
    runner = make_runner(
        profile_store, full_workflow(), screen_with_button(button_patch), dry_run=True
    )
    result = runner.run()
    assert (runner.run_dir / "debug_ok_button.png").is_file()
    assert result.ok


def test_normal_run_clicks_pastes_and_logs(profile_store, button_target, button_patch):
    spy = SpyController()
    runner = make_runner(
        profile_store,
        full_workflow(),
        screen_with_button(button_patch),
        dry_run=False,
        controller=spy,
    )
    result = runner.run()

    assert result.status == STATUS_SUCCESS
    names = [c[0] for c in spy.calls]
    assert names == ["move_to", "click", "paste_text", "hotkey"]

    # FakeScreen has scale 1.0, so logical coords equal template-match pixels.
    expected_x = BUTTON_POS[0] + button_patch.shape[1] // 2
    expected_y = BUTTON_POS[1] + button_patch.shape[0] // 2
    _, click_x, click_y = spy.calls[1]
    assert abs(click_x - expected_x) <= 3
    assert abs(click_y - expected_y) <= 3
    assert ("paste_text", "hello") in spy.calls
    assert ("hotkey", "ctrl", "s") in spy.calls

    events = read_run_log(result.run_dir)
    click_events = [e for e in events if e["type"] == "step_finished" and e["step_id"] == "click"]
    assert click_events[0]["status"] == STATUS_SUCCESS
    assert click_events[0]["confidence"] > 0.9
    assert "Clicked target at" in click_events[0]["message"]


def test_failure_stops_run_and_saves_screenshot(profile_store, button_target):
    # Screen WITHOUT the button: click must fail, following steps must not run.
    empty_screen = FakeScreen(make_background(seed=3))
    spy = SpyController()
    workflow = make_workflow(
        [
            WorkflowStep(
                id="click", action="click_target", target_id="ok_button", timeout_seconds=0.3
            ),
            WorkflowStep(id="paste", action="paste_text", text="never"),
        ]
    )
    runner = make_runner(profile_store, workflow, empty_screen, dry_run=False, controller=spy)
    result = runner.run()

    assert result.status == STATUS_FAILURE
    assert spy.calls == []  # no click, and paste never ran
    statuses = {r.step_id: r.status for r in result.step_results}
    assert statuses == {"click": STATUS_FAILURE, "paste": STATUS_SKIPPED}

    click_result = result.step_results[0]
    assert "not found" in click_result.message
    assert click_result.error
    assert (runner.run_dir / "failure_click.png").is_file()

    events = read_run_log(result.run_dir)
    failure_events = [e for e in events if e.get("status") == STATUS_FAILURE]
    assert failure_events and failure_events[0]["confidence"] is not None


def test_continue_on_failure(profile_store, button_target):
    empty_screen = FakeScreen(make_background(seed=3))
    workflow = make_workflow(
        [
            WorkflowStep(
                id="find", action="find_target", target_id="ok_button", continue_on_failure=True
            ),
            WorkflowStep(id="note", action="notify", message="still here"),
        ]
    )
    runner = make_runner(profile_store, workflow, empty_screen, dry_run=False)
    result = runner.run()
    statuses = [r.status for r in result.step_results]
    assert statuses == [STATUS_FAILURE, STATUS_SUCCESS]
    assert result.status == STATUS_FAILURE  # run still reports the failure


def test_stop_action_skips_remaining_steps(profile_store, button_target, button_patch):
    workflow = make_workflow(
        [
            WorkflowStep(id="note", action="notify", message="hi"),
            WorkflowStep(id="halt", action="stop"),
            WorkflowStep(id="paste", action="paste_text", text="never"),
        ]
    )
    spy = SpyController()
    runner = make_runner(
        profile_store,
        workflow,
        screen_with_button(button_patch),
        dry_run=False,
        controller=spy,
    )
    result = runner.run()
    assert result.status == STATUS_SUCCESS
    statuses = {r.step_id: r.status for r in result.step_results}
    assert statuses["halt"] == STATUS_SUCCESS
    assert statuses["paste"] == STATUS_SKIPPED
    assert spy.calls == []


def test_validation_failure_blocks_run(profile_store, button_target, button_patch):
    workflow = make_workflow(
        [WorkflowStep(id="click", action="click_target", target_id="no_such_target")]
    )
    spy = SpyController()
    runner = make_runner(
        profile_store,
        workflow,
        screen_with_button(button_patch),
        dry_run=False,
        controller=spy,
    )
    result = runner.run()
    assert result.status == STATUS_FAILURE
    assert "unknown target" in result.message
    assert result.step_results == []
    assert spy.calls == []


def test_pre_cancelled_run_executes_nothing(profile_store, button_target, button_patch):
    cancel_event = threading.Event()
    cancel_event.set()
    spy = SpyController()
    runner = make_runner(
        profile_store,
        full_workflow(),
        screen_with_button(button_patch),
        dry_run=False,
        controller=spy,
        cancel_event=cancel_event,
    )
    result = runner.run()
    assert result.status == STATUS_CANCELLED
    assert spy.calls == []
    assert all(r.status in (STATUS_CANCELLED, STATUS_SKIPPED) for r in result.step_results)


def test_wait_for_any_target(profile_store, button_target, button_patch):
    workflow = make_workflow(
        [
            WorkflowStep(
                id="any", action="wait_for_any_target", target_ids=["ok_button"], timeout_seconds=2
            ),
        ]
    )
    runner = make_runner(profile_store, workflow, screen_with_button(button_patch), dry_run=True)
    result = runner.run()
    assert result.ok
    assert result.step_results[0].target_id == "ok_button"
