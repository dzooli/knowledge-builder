# Contributing Guide

Thanks for your interest in improving Knowledge Builder!

## Workflow

1. Fork & branch from `master` (use descriptive branch names).
2. Keep changes focused; one feature or fix per PR.
3. Add/adjust tests (`pytest`) for any behavior changes.
4. Run local quality gates before pushing.

## Development Setup

```bash
uv sync --dev
uv run pytest -q
uv run mkdocs serve  # optional docs preview
uv run pre-commit install  # install git hooks (monolith guard, etc.)
```

## Code Standards

* Python 3.13, type hints required.
* Keep function length < 30 lines, cognitive complexity < 15.
* Prefer helper abstractions over duplication.
* Use Pydantic models or slotted dataclasses for structured data.
* Logging via `loguru` only; include meaningful context.
* Pre-commit monolith guard blocks adding Python modules > 800 lines (excluding tests & special cases). Adjust thresholds in `scripts/monolith_guard.py` if justified.

## Architecture Docs & ADRs

* Significant design changes require an ADR.
* Create one with:
  

```bash
  uv run python scripts/new_adr.py "Your Decision Title"
  ```

* Fill sections (Context, Decision, Options, Rationale, Consequences) then mark status.

## Agent Orchestrator

* Staged pipeline (see `docs/agent_orchestrator.md`).
* New semantic behavior: prefer enhancing existing stages before adding a new one.
* Tool invocations must go through `_invoke_tool` for consistent logging.

## Tests

* Unit tests for helpers and edge cases.
* Runtime tests for orchestration fallbacks & timeout path.
* Add regression tests for any fixed bug.

## Docs

* Add or update pages under `docs/`.
* API docs are generated via mkdocstrings; ensure public APIs have docstrings.
* For versioned release docs: use GitHub workflow `docs-version` with input tag.

## Commit Messages

* Conventional style helpful but not enforced: `feat:`,   `fix:`,   `docs:`,   `refactor:`, etc.

## Pull Requests

* Link related ADR or issue.
* Provide a short summary of approach & trade-offs.
* Ensure CI (tests + docs build) passes.

## License

By contributing you agree to license your work under the project’s MIT License.

Happy building!
