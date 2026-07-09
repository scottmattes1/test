"""Filesystem persistence: workspace, profiles, targets, workflows, run folders.

Layout (all local, no cloud):

    <workspace root>/
      profiles/
        <profile_id>/
          app_profile.yaml
          targets/
            <target_id>.yaml
            <target_id>.png
          workflows/
            <workflow_id>.yaml
          runs/
            2026-07-09_14-31-22/
              run.jsonl
              failure_<step_id>.png
              debug_<target_id>.png

The default workspace root is ``~/LocalRPAStudio`` and can be overridden with
the ``LOCAL_RPA_STUDIO_HOME`` environment variable or the ``--workspace`` CLI
flag. No absolute paths are stored inside YAML files; template paths are
relative to the profile directory so profiles can be copied/shared freely.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

from .models import (
    AppProfile,
    Target,
    Workflow,
    now_iso,
    profile_from_dict,
    profile_to_dict,
    target_from_dict,
    target_to_dict,
    workflow_from_dict,
    workflow_to_dict,
)

PROFILE_FILE = "app_profile.yaml"
TARGETS_DIR = "targets"
WORKFLOWS_DIR = "workflows"
RUNS_DIR = "runs"


class StorageError(RuntimeError):
    """Raised for any load/save problem, with a user-facing message."""


def default_workspace_root() -> Path:
    env = os.environ.get("LOCAL_RPA_STUDIO_HOME")
    if env:
        return Path(env).expanduser()
    return Path.home() / "LocalRPAStudio"


def _read_yaml(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except FileNotFoundError as exc:
        raise StorageError(f"File not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise StorageError(f"Invalid YAML in {path}: {exc}") from exc
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise StorageError(f"Expected a YAML mapping in {path}, got {type(data).__name__}")
    return data


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)


class ProfileStore:
    """All reads/writes for a single app profile directory."""

    def __init__(self, profile_dir: Path):
        self.profile_dir = Path(profile_dir)

    # -- paths --------------------------------------------------------------

    @property
    def profile_id(self) -> str:
        return self.profile_dir.name

    @property
    def targets_dir(self) -> Path:
        return self.profile_dir / TARGETS_DIR

    @property
    def workflows_dir(self) -> Path:
        return self.profile_dir / WORKFLOWS_DIR

    @property
    def runs_dir(self) -> Path:
        return self.profile_dir / RUNS_DIR

    def ensure_dirs(self) -> None:
        for path in (self.profile_dir, self.targets_dir, self.workflows_dir, self.runs_dir):
            path.mkdir(parents=True, exist_ok=True)

    # -- profile ------------------------------------------------------------

    def load_profile(self) -> AppProfile:
        return profile_from_dict(_read_yaml(self.profile_dir / PROFILE_FILE))

    def save_profile(self, profile: AppProfile) -> None:
        profile.updated_at = now_iso()
        _write_yaml(self.profile_dir / PROFILE_FILE, profile_to_dict(profile))

    # -- targets ------------------------------------------------------------

    def list_target_ids(self) -> list[str]:
        if not self.targets_dir.is_dir():
            return []
        return sorted(p.stem for p in self.targets_dir.glob("*.yaml"))

    def list_targets(self) -> list[Target]:
        return [self.load_target(target_id) for target_id in self.list_target_ids()]

    def load_target(self, target_id: str) -> Target:
        return target_from_dict(_read_yaml(self.targets_dir / f"{target_id}.yaml"))

    def save_target(self, target: Target, image_bgr: np.ndarray | None = None) -> None:
        """Save target metadata, and its template image when provided."""
        self.ensure_dirs()
        if not target.template_path:
            target.template_path = f"{TARGETS_DIR}/{target.id}.png"
        if image_bgr is not None:
            template_path = self.template_abspath(target)
            template_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(template_path), image_bgr):
                raise StorageError(f"Could not write template image: {template_path}")
        target.updated_at = now_iso()
        _write_yaml(self.targets_dir / f"{target.id}.yaml", target_to_dict(target))

    def delete_target(self, target_id: str) -> None:
        try:
            target = self.load_target(target_id)
            self.template_abspath(target).unlink(missing_ok=True)
        except StorageError:
            pass
        (self.targets_dir / f"{target_id}.yaml").unlink(missing_ok=True)

    def template_abspath(self, target: Target) -> Path:
        return self.profile_dir / target.template_path

    def load_template(self, target: Target) -> np.ndarray:
        path = self.template_abspath(target)
        if not path.is_file():
            raise StorageError(
                f"Template image for target '{target.id}' is missing: {path}. "
                "Re-capture the target to fix this."
            )
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise StorageError(f"Template image for target '{target.id}' could not be read: {path}")
        return image

    # -- workflows ------------------------------------------------------------

    def list_workflow_ids(self) -> list[str]:
        if not self.workflows_dir.is_dir():
            return []
        return sorted(p.stem for p in self.workflows_dir.glob("*.yaml"))

    def workflow_path(self, workflow_id: str) -> Path:
        return self.workflows_dir / f"{workflow_id}.yaml"

    def load_workflow(self, workflow_id: str) -> Workflow:
        return workflow_from_dict(_read_yaml(self.workflow_path(workflow_id)))

    def save_workflow(self, workflow: Workflow) -> None:
        self.ensure_dirs()
        _write_yaml(self.workflow_path(workflow.id), workflow_to_dict(workflow))

    def read_workflow_text(self, workflow_id: str) -> str:
        path = self.workflow_path(workflow_id)
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise StorageError(f"Workflow file not found: {path}") from exc

    def write_workflow_text(self, workflow_id: str, text: str) -> None:
        """Write raw YAML text (preserves comments/formatting from the editor)."""
        self.ensure_dirs()
        self.workflow_path(workflow_id).write_text(text, encoding="utf-8")

    def delete_workflow(self, workflow_id: str) -> None:
        self.workflow_path(workflow_id).unlink(missing_ok=True)

    # -- runs ------------------------------------------------------------------

    def new_run_dir(self) -> Path:
        self.ensure_dirs()
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = self.runs_dir / stamp
        counter = 2
        while run_dir.exists():
            run_dir = self.runs_dir / f"{stamp}_{counter}"
            counter += 1
        run_dir.mkdir(parents=True)
        return run_dir


class Workspace:
    """The root folder holding all profiles."""

    def __init__(self, root: Path | str | None = None):
        self.root = Path(root).expanduser() if root else default_workspace_root()

    @property
    def profiles_dir(self) -> Path:
        return self.root / "profiles"

    def ensure(self) -> None:
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

    def list_profile_ids(self) -> list[str]:
        if not self.profiles_dir.is_dir():
            return []
        return sorted(p.name for p in self.profiles_dir.iterdir() if (p / PROFILE_FILE).is_file())

    def list_profiles(self) -> list[AppProfile]:
        return [self.profile_store(pid).load_profile() for pid in self.list_profile_ids()]

    def profile_store(self, profile_id: str) -> ProfileStore:
        profile_dir = self.profiles_dir / profile_id
        if not (profile_dir / PROFILE_FILE).is_file():
            raise StorageError(
                f"Profile '{profile_id}' not found in {self.profiles_dir} (no {PROFILE_FILE})."
            )
        return ProfileStore(profile_dir)

    def create_profile(
        self, display_name: str, profile_id: str | None = None, description: str = ""
    ) -> ProfileStore:
        from .models import slugify

        profile_id = profile_id or slugify(display_name)
        profile_dir = self.profiles_dir / profile_id
        if (profile_dir / PROFILE_FILE).exists():
            raise StorageError(f"A profile with id '{profile_id}' already exists.")
        store = ProfileStore(profile_dir)
        store.ensure_dirs()
        store.save_profile(
            AppProfile(id=profile_id, display_name=display_name, description=description)
        )
        return store
