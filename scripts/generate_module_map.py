#!/usr/bin/env python
"""Generate (and optionally update) the Module Map section in README.

Usage (dry run print table to stdout):
    uv run python scripts/generate_module_map.py --print

Usage (in-place README update between markers):
    uv run python scripts/generate_module_map.py --write

Logic:
1. Scan `importer/src` recursively for Python files (excluding __pycache__, tests, compiled, notebooks).
2. Derive an *Area* classification from the relative path segment (heuristic mapping).
3. Extract first non-empty line of the module-level docstring (if present) as Responsibility; fallback to a short heuristic phrase.
4. Emit a Markdown table with header lines wrapped by markers `<!-- MODULE_MAP_START -->` / `<!-- MODULE_MAP_END -->`.
5. When `--write` is used, replace existing table inside README markers, preserving surrounding content.

Design Principles:
- Pure function decomposition (easier testing)
- No external dependencies beyond stdlib
- Deterministic ordering (Area order, then path)

Exit Codes:
- 0 success
- 1 write marker not found (when --write)
- 2 unexpected exception
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
IMPORTER_SRC = REPO_ROOT / "importer" / "src"
README_PATH = REPO_ROOT / "README.md"
MARKER_START = "<!-- MODULE_MAP_START -->"
MARKER_END = "<!-- MODULE_MAP_END -->"

AREA_ORDER = [
    "Entry",
    "Configuration",
    "Data Models",
    "Connectors",
    "Processing",
    "Services",
    "Utilities",
    "Scripts",
]


@dataclass(slots=True)
class ModuleRow:
    area: str
    rel_path: str
    responsibility: str


def iter_python_files(base: Path) -> Iterable[Path]:
    for path in base.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        if path.name.startswith("."):
            continue
        yield path


def classify_area(path: Path) -> str:
    parts = path.parts
    # Look for known folder names
    if "services" in parts:
        return "Services"
    if "connectors" in parts:
        return "Connectors"
    if "processing" in parts:
        return "Processing"
    if "utils" in parts:
        return "Utilities"
    if path.name == "config.py":
        return "Configuration"
    if path.name == "models.py":
        return "Data Models"
    if path.name == "main.py":
        return "Entry"
    # Fallback: scripts outside importer
    if REPO_ROOT / "scripts" in path.parents:
        return "Scripts"
    return "Utilities"


def extract_docstring_first_line(py_file: Path) -> Optional[str]:
    try:
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):  # skip problematic
        return None
    doc = ast.get_docstring(tree)
    if not doc:
        return None
    for line in doc.splitlines():
        line = line.strip()
        if line:
            return line[:160]
    return None


def responsibility_for(path: Path) -> str:
    doc_line = extract_docstring_first_line(path)
    if doc_line:
        return doc_line
    # Heuristic fallback
    name = path.stem.replace("_", " ")
    return f"{name.capitalize()} module"


def collect_rows() -> List[ModuleRow]:
    rows: List[ModuleRow] = []
    for file in iter_python_files(IMPORTER_SRC):
        area = classify_area(file)
        rel_path = file.relative_to(REPO_ROOT).as_posix()
        responsibility = responsibility_for(file)
        rows.append(ModuleRow(area, rel_path, responsibility))
    # Deterministic ordering: AREA_ORDER then alphabetical path
    area_rank = {a: i for i, a in enumerate(AREA_ORDER)}
    rows.sort(key=lambda r: (area_rank.get(r.area, 999), r.rel_path))
    return rows


def build_table(rows: List[ModuleRow]) -> str:
    header = "| Area | Path / Module | Responsibility |"
    sep = "|------|---------------|----------------|"
    lines = [header, sep]
    for r in rows:
        lines.append(f"| {r.area} | `{r.rel_path}` | {r.responsibility} |")
    return "\n".join(lines)


def replace_in_readme(table: str) -> Tuple[bool, str]:
    content = README_PATH.read_text(encoding="utf-8")
    if MARKER_START not in content or MARKER_END not in content:
        return False, content
    pre, rest = content.split(MARKER_START, 1)
    mid, post = rest.split(MARKER_END, 1)
    new_mid = f"{MARKER_START}\n{table}\n{MARKER_END}"
    return True, pre + new_mid + post


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Module Map table")
    parser.add_argument("--print", action="store_true", help="Print table to stdout")
    parser.add_argument(
        "--write", action="store_true", help="Write table into README between markers"
    )
    args = parser.parse_args()

    rows = collect_rows()
    table = build_table(rows)

    if args.print or not (
        args.print or args.write
    ):  # default to print if nothing specified
        print(table)
    if args.write:
        ok, new_content = replace_in_readme(table)
        if not ok:
            print("[module-map] markers not found in README; aborting", flush=True)
            return 1
        README_PATH.write_text(new_content, encoding="utf-8")
        print("[module-map] README updated", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI tool
    raise SystemExit(main())
