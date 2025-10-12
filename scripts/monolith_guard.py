#!/usr/bin/env python
"""Pre-commit hook script to block overly large Python modules ("monoliths").

Policy:
- Fail if any newly added or modified *.py file (excluding tests, migrations, __init__, and scripts/monolith_guard.py itself)
  exceeds MAX_LINES or MAX_COMPLEXITY placeholders.
- Provides a gentle warning threshold as well.

Complexity heuristic kept intentionally simple: counts 'def ' occurrences; if functions
are too few relative to lines it may indicate long functions; this is a heuristic only.

Adjust thresholds below as architecture evolves.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

MAX_LINES = 800  # hard cap for monolith detection
WARN_LINES = 600  # warn above this

# crude heuristic: minimum number of functions per 200 lines
MIN_FUNCS_PER_200 = 3

EXEMPT_SUBSTRINGS = [
    "tests/",
    "__init__.py",
    "scripts/monolith_guard.py",
]


def is_exempt(path: str) -> bool:
    return any(token in path.replace("\\", "/") for token in EXEMPT_SUBSTRINGS)


def get_changed_python_files() -> list[Path]:
    cmd = ["git", "diff", "--cached", "--name-only", "--diff-filter=AM"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    files = []
    for line in res.stdout.splitlines():
        if line.endswith(".py") and not is_exempt(line):
            files.append(Path(line))
    return files


def analyze_file(path: Path) -> tuple[int, int]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return 0, 0
    lines = text.count("\n") + 1
    funcs = text.count("\ndef ") + (1 if text.startswith("def ") else 0)
    return lines, funcs


def main() -> int:
    offenders = []
    warnings = []
    for file in get_changed_python_files():
        lines, funcs = analyze_file(file)
        if lines == 0:
            continue
        if lines > MAX_LINES:
            offenders.append((file, lines, funcs))
        elif lines > WARN_LINES:
            warnings.append((file, lines, funcs))
        # Heuristic complexity: lines per function
        if funcs > 0:
            if (lines / 200) * MIN_FUNCS_PER_200 > funcs and lines > WARN_LINES:
                warnings.append((file, lines, funcs))
    if offenders:
        print("Monolith guard: blocking commit - large modules detected:")
        for path, line_count, func_count in offenders:
            print(f"  - {path} : {line_count} lines, functions={func_count}")
        print(
            "Set smaller modules, refactor, or adjust thresholds in scripts/monolith_guard.py"
        )
        return 1
    if warnings:
        print("Monolith guard: warnings (consider refactoring):")
        seen = set()
        for path, line_count, func_count in warnings:
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            print(f"  - {path} : {line_count} lines, functions={func_count}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
