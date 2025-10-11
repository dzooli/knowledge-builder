# ADR 0001: Staged Agent Orchestrator Refactor

Date: 2025-10-11
Status: Accepted

## Context

The original `agent_orchestrator.py` contained a large monolithic method with duplicated logic (evidence creation, tool invocation patterns, heuristic fallbacks) that made extension risky and testing difficult. Timeout handling and fallback guarantees were implicit, and logging repeated verbose formatting, reducing clarity.

## Decision

Refactor the orchestrator into a 5-stage pipeline with layered fallbacks and supporting helper abstractions:
1. Primary Agent Run (timeout bound)
2. Execute Parsed Tool Calls
3. Heuristic Augmentation
4. Forced Minimal Write
5. Finalize

Introduce helper methods for evidence naming, entity construction, tool invocation, heuristic extraction, linking, and observation insertion. Expose thin public wrappers for testing. Centralize logging and exception handling policies. Guarantee that every chunk produces at least one Evidence node and semantic content.

## Alternatives Considered

* Keep monolith + incrementally patch: rejected (complexity & duplication persist)
* Full DSL for tool planning: overkill for current scale, increases cognitive load
* Pure heuristic pipeline (no agent): loses semantic richness and adaptive extraction

## Consequences

Positive:
* Lower cognitive complexity (<30 lines per function target maintained)
* Easier unit & runtime testing (public wrappers)
* Deterministic fallback behavior, fewer silent failures
* Clear extension seams (add stage vs. extend helper)

Negative / Trade-offs:
* Slight overhead of additional function indirections
* More files/doc sections to maintain

## Implementation Notes

* `_run_with_timeout` replaces earlier sync wrapper
* `_invoke_tool` centralizes success detection & logging
* Heuristic extraction limited to proper-case tokens to remain lightweight
* Fallback ladder documented in diagram; early sufficiency check prevents redundant writes

## Metrics / Validation

* Tests cover helpers, timeout path, and forced fallback
* Manual log inspection shows truncated raw LLM output and uniform tool logs

## Future Work

* Adaptive timeout per chunk size & historical complexity
* Metric emission for stage durations & fallback frequency
* Re-enrichment background job for minimal Evidence nodes

## Status

Implemented and documented (see `docs/agent_orchestrator.md` ).
