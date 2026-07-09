"""Command-line entry point: ``local-rpa-studio`` / ``python -m local_rpa_studio.main``."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="local-rpa-studio",
        description="Local-first desktop RPA / workflow automation studio.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help=(
            "Workspace folder holding profiles (default: ~/LocalRPAStudio, "
            "or $LOCAL_RPA_STUDIO_HOME)"
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args(argv)

    from .app import run_app  # imported lazily so --help/--version work without Qt

    return run_app(args.workspace)


if __name__ == "__main__":
    raise SystemExit(main())
