"""CLI wrappers for documentation tasks.

These lightweight wrappers allow invoking docs commands via
`uv run docs-build`, `uv run docs-serve`, etc., using PEP 621 console scripts.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Sequence


def _run(cmd: Sequence[str]) -> int:
    print(f"[docs-cli] executing: {' '.join(cmd)}")
    try:
        return subprocess.call(cmd)
    except FileNotFoundError as e:  # pragma: no cover
        print(f"Command not found: {cmd[0]} ({e})", file=sys.stderr)
        return 127


def serve() -> int:
    return _run([sys.executable, "-m", "mkdocs", "serve"])


def build() -> int:
    return _run([sys.executable, "-m", "mkdocs", "build", "--strict"])


def deploy_version() -> int:
    if len(sys.argv) < 2:
        print("Usage: uv run docs-deploy-version <version> [alias...]", file=sys.stderr)
        return 2
    version = sys.argv[1]
    aliases = sys.argv[2:]
    cmd = [
        sys.executable,
        "-m",
        "mike",
        "deploy",
        "--update-aliases",
        version,
    ] + aliases
    return _run(cmd)


def set_default() -> int:
    if len(sys.argv) < 2:
        print("Usage: uv run docs-set-default <alias>", file=sys.stderr)
        return 2
    alias = sys.argv[1]
    cmd = [sys.executable, "-m", "mike", "set-default", alias]
    return _run(cmd)


if __name__ == "__main__":  # pragma: no cover
    # Allow direct execution for quick manual testing; default to build.
    sys.exit(build())
