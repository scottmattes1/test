# AGENTS.md — working on Local RPA Studio

Guidance for AI agents and new contributors.

## Commands

```bash
pip install -e ".[dev]"       # full install (needs a display for opencv/pyautogui)
pytest                        # run tests (headless-safe)
ruff check src tests          # lint
ruff format src tests         # format
local-rpa-studio              # launch the GUI
rpa-demo-app                  # launch the demo target app
```

Headless environments (CI, containers): install `opencv-python-headless`
instead of `opencv-python`, and skip `PySide6`/`PyAutoGUI` — the test suite
does not import them.

## Architecture invariants — keep these true

1. **Lazy GUI-stack imports.** `pyautogui` and `pyperclip` are imported
   *inside* functions/methods (see `screen.py`, `executor/controls.py`), never
   at module top level. This keeps every non-`ui/` module importable headless,
   which is what makes the test suite runnable anywhere.
2. **The UI is a thin shell.** All behaviour lives in `models`, `storage`,
   `locator`, and `executor`; `ui/` only wires widgets to those layers.
   Nothing outside `ui/`, `app.py`, and `demo_app.py` may import PySide6.
3. **Dependency injection for hardware.** The runner takes `screen`
   (ScreenSource) and `controller` (input) arguments. Tests pass `FakeScreen`
   and `SpyController` (see `tests/conftest.py`); production code passes
   nothing and gets the real ones. Preserve this seam.
4. **Dry-run guarantee.** In dry-run mode no handler may move the mouse,
   click, type, paste, or press hotkeys. Screenshots/matching are allowed.
   `test_dry_run_performs_no_input_actions` enforces this — extend it when
   adding actions.
5. **Locators are pluggable.** New locator methods (accessibility, OCR,
   Playwright) subclass `locator.base.Locator`, get `@register_locator`, and
   are selected per-target via `locator.method` in YAML. Don't special-case
   locator types in the executor.
6. **Confidence gating.** `MatchResult.found` is the only thing that permits a
   click; it is computed as `confidence >= target.min_confidence`. Never click
   on a not-found match.
7. **Files are the source of truth.** Profiles/targets/workflows are plain
   YAML + PNG under the workspace; no databases, no hidden state, no absolute
   paths inside YAML (template paths are profile-relative). Run logs are
   append-only JSONL.
8. **Local-first.** No network calls, no cloud services, no auth, no LLM
   agents, no telemetry.

## Adding a new step action

1. Add the name to `VALID_ACTIONS` in `models.py` and add validation rules in
   `validate_step` if it needs specific fields.
2. Write a handler `action_<name>(ctx, step) -> StepOutcome` in
   `executor/actions.py` and register it in `ACTION_HANDLERS`. Respect
   `ctx.dry_run` and use `ctx.sleep()`/`ctx.check_cancel()` for waits.
3. Document it in the `NEW_WORKFLOW_TEMPLATE` comment
   (`ui/workflow_editor.py`) and the README.
4. Add a runner test in `tests/test_runner.py` (both dry and normal mode if
   it performs input).

## Conventions

- Python 3.11+, ruff-formatted, line length 100, type hints everywhere.
- User-facing error messages must say what to *do* (e.g. "re-capture the
  target", "check Screen Recording permission"), not just what broke.
- macOS security is respected, never bypassed: detect failures, explain the
  System Settings fix (`permissions.py`).
- Keep tests deterministic: synthetic images from `tests/conftest.py`, no
  sleeps longer than ~0.3s, no real screen/input.
