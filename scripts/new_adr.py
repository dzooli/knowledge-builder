#!/usr/bin/env python3
"""Automate creation of a new Architecture Decision Record (ADR).

Usage:
    uv run python scripts/new_adr.py "Title of Decision"

Creates: docs/adr/NNNN-kebab-title.md with a standard template.
"""

from __future__ import annotations

import sys
import re
from pathlib import Path
from datetime import date

ADR_DIR = Path("docs/adr")
TEMPLATE = """# ADR {number}: {title}

Date: {today}
Status: Proposed

## Context
Describe the problem, forces, and background leading to this decision.

## Decision
State the decision in a single concise paragraph.

## Options Considered
- Option A
- Option B
- Option C

## Rationale
Why this option? Reference trade-offs, constraints, and evaluation criteria.

## Consequences
Positive:
- 
Negative:
- 

## Implementation Notes
(Planned high-level steps or references to PRs / issues.)

## Related
(Link to related ADRs, issues, or external docs.)

## Status
(Status progression: Proposed -> Accepted / Rejected / Superseded.)
"""


def slugify(title: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", title.strip().lower()).strip("-")
    return slug or "decision"


def next_number() -> int:
    ADR_DIR.mkdir(parents=True, exist_ok=True)
    numbers = []
    for p in ADR_DIR.glob("[0-9][0-9][0-9][0-9]-*.md"):
        try:
            numbers.append(int(p.name.split("-", 1)[0]))
        except ValueError:
            continue
    return (max(numbers) + 1) if numbers else 1


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: new_adr.py 'Title of Decision'", file=sys.stderr)
        return 1
    title = argv[1]
    number = next_number()
    slug = slugify(title)
    filename = f"{number:04d}-{slug}.md"
    path = ADR_DIR / filename
    if path.exists():
        print(f"ADR already exists: {path}", file=sys.stderr)
        return 2
    content = TEMPLATE.format(
        number=f"{number:04d}", title=title, today=date.today().isoformat()
    )
    path.write_text(content, encoding="utf-8")
    print(f"Created {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv))
