# Local RPA Studio

A **local-first desktop RPA / workflow automation studio for macOS**, written in
Python. Capture visual targets (buttons, labels, fields) as screenshots, build
simple YAML workflows out of them, and run those workflows to click, paste
text, wait for screen elements, and log every step — all on your own machine,
with no cloud dependency, no accounts, and no telemetry.

## What the MVP does

- **App profiles** — one folder per app or website you automate, holding its
  targets, workflows, and run logs.
- **Target capture** — freeze the screen, drag a rectangle around a UI
  element, and save it as a PNG template plus YAML metadata (id, kind,
  minimum match confidence, click offset).
- **Visual matching** — OpenCV template matching (`cv2.TM_CCOEFF_NORMED`, on
  grayscale) with simple multi-scale search to cope with Retina/DPI
  differences.
- **Workflows** — YAML files with steps: `wait_seconds`, `find_target`,
  `wait_for_target`, `wait_for_any_target`, `click_target`, `paste_text`,
  `copy_clipboard`, `hotkey`, `screenshot`, `notify`, `stop`.
- **Runner** — normal runs and **dry runs** (match targets and log what would
  happen, but never touch mouse/keyboard/clipboard), per-step timeouts,
  stop-on-failure, cancellation from the UI, and a pre-run countdown so you
  can focus the target app.
- **Logging** — one folder per run with a `run.jsonl` event log, failure
  screenshots, and optional debug images with match bounding boxes drawn.

## What it does *not* do yet

- No macOS Accessibility / OCR / browser-DOM locators (vision-only for now).
- No conditional branching, loops, or variables in workflows.
- No window-scoped search regions — matching runs on the full primary screen.
- No packaged `.app`; you run it from a Python environment.
- It cannot promise pixel-perfect reliability across every app, theme, or
  display configuration — always dry-run first.

## Installation

Requires **Python 3.11+** on macOS.

```bash
git clone <this repo>
cd <repo>
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

> Headless CI note: `opencv-python` and `PyAutoGUI` need a display. For
> running the test suite on a headless machine, install
> `opencv-python-headless` instead of `opencv-python`; the tests never touch
> the real screen or input devices.

## Running the app

```bash
local-rpa-studio                 # uses ~/LocalRPAStudio as the workspace
local-rpa-studio --workspace ~/my-rpa-workspace
```

You can also set `LOCAL_RPA_STUDIO_HOME` to change the default workspace.

To experiment safely, start the bundled demo target app in a second terminal:

```bash
rpa-demo-app
```

## macOS permissions

macOS gates screen capture and input synthesis behind per-app permissions.
Open **System Settings → Privacy & Security** and enable your terminal (or
Python) under:

- **Screen Recording** — required for screenshots and target matching.
  Without it, screenshots come back black; the app detects this and tells you.
- **Accessibility** — required for mouse clicks and keyboard/paste actions.
- **Input Monitoring** — only if keyboard actions still fail after enabling
  Accessibility.

After changing a permission, fully quit and restart the app. Use the
**Check Permissions** button in the toolbar to verify screen capture works.
Local RPA Studio never tries to bypass macOS security; it only detects the
common failure modes and explains the fix.

Safety valve: PyAutoGUI's fail-safe is left enabled — slam the mouse into the
**top-left corner of the screen** to abort a run instantly.

## Quick start

### 1. Create a profile

Click **New Profile…**, name it after the app you're automating (e.g.
"QuickBooks"). A folder appears under `<workspace>/profiles/<profile_id>/`.

### 2. Capture a target

1. Bring the app you want to automate on screen.
2. Click **Capture Target…** — the studio window hides, takes a screenshot,
   and shows it full-screen (frozen).
3. Drag a rectangle around the element (a button, a label…). Esc cancels.
4. Fill in the id / display name / kind / minimum confidence. Done — the crop
   is saved as `targets/<id>.png` with metadata in `targets/<id>.yaml`.

Tips: capture tightly around the element, avoid regions that change (cursors,
hover highlights), and re-capture if you change screen resolution or theme.

### 3. Create a workflow

Click **New Workflow…** — a template YAML opens in the editor. Edit the steps,
point `target_id` at your captured targets, then **Validate** and **Save**:

```yaml
id: demo_submit_form
display_name: Demo Submit Form
steps:
  - id: wait_for_button
    action: wait_for_target
    target_id: submit_button
    timeout_seconds: 10
  - id: click_button
    action: click_target
    target_id: submit_button
    timeout_seconds: 10
  - id: paste_value
    action: paste_text
    text: "Hello from Local RPA Studio"
  - id: final_wait
    action: wait_seconds
    seconds: 1
```

Step options: every step takes `timeout_seconds` (for the target-based
actions) and `continue_on_failure: true` to keep going after a failure.
Workflow-level options: `countdown_seconds` and `stop_on_failure`.

### 4. Dry run

Click **Dry Run**. The runner matches every target and logs what it *would*
do — including click coordinates and match confidence — without moving the
mouse or typing. Debug images with bounding boxes are saved automatically for
dry runs (checkbox enables them for normal runs too).

### 5. Run for real

Click **▶ Run**. A 3-second countdown gives you time to focus the target app,
then steps execute. **Cancel Run** stops after the current step; the top-left
screen corner (PyAutoGUI fail-safe) aborts immediately. The runner never
clicks when match confidence is below the target's `min_confidence`.

### 6. Inspect logs

Every run writes to `<profile>/runs/<timestamp>/`:

```
runs/2026-07-09_14-31-22/
  run.jsonl               # one JSON event per line
  failure_click_button.png  # screenshot taken when a step failed
  debug_submit_button.png   # match bounding box (dry runs / opt-in)
```

Example event line:

```json
{"timestamp": "2026-07-09T14:31:22", "type": "step_finished", "workflow_id": "demo_submit_form",
 "step_id": "click_button", "action": "click_target", "status": "success",
 "target_id": "submit_button", "confidence": 0.92, "message": "Clicked target at x=1105, y=761"}
```

**Open Runs Folder** in the UI takes you straight there.

## Example profile

See [`examples/sample_profile/`](examples/sample_profile/) for a ready-made
profile targeting the bundled demo app, including copy instructions. It works
against any visible UI element — a button in your browser works just as well.

## Project layout

```
src/local_rpa_studio/
  models.py          # dataclasses: AppProfile, Target, Workflow, Step/Run results
  storage.py         # workspace/profile folders, YAML + PNG persistence
  logging_utils.py   # JSONL run logger
  screen.py          # screenshot capture + Retina scale conversion
  permissions.py     # macOS permission detection & hints
  locator/           # pluggable locators; vision.py = OpenCV template matching
  executor/          # actions.py (step handlers), runner.py, controls.py (input)
  ui/                # PySide6 main window, capture overlay, workflow editor
  demo_app.py        # tiny practice target app
```

## Tests

```bash
pytest
ruff check src tests
```

The tests use synthetic images, a fake screen, and a spy input controller — no
display, permissions, or real input needed. They cover YAML round-trips,
workflow validation, the vision locator (including multi-scale), and runner
behaviour (dry-run never touches input, failures save screenshots, etc.).

## Known limitations

- Template matching is sensitive to theme changes, font rendering, and
  overlapping windows; multi-scale matching helps with DPI but not rotation
  or heavy restyling.
- Only the primary screen is searched; multi-monitor setups are untested.
- `search_region` is stored but only `active_screen` (full screen) is
  implemented.
- Clipboard-based paste replaces whatever was on your clipboard.
- The runner clicks wherever the match is — if the screen changes between
  match and click, the click can land wrong. Keep machines idle during runs.

## Roadmap

- macOS Accessibility API locator (resolution-independent, more reliable)
- OCR/text locator (find elements by their visible text)
- Playwright web mode (DOM locators for browser workflows)
- Conditional branching (`if target visible then … else …`)
- CSV/Excel-driven form filling (row-per-run batches)
- Human approval checkpoints mid-workflow
- Packaged `.app` build (PyInstaller/briefcase)
- Signed & notarized macOS release
