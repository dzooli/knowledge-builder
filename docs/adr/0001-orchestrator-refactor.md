# ADR 0001: Staged Agent Orchestrator Refactor

Date: 2025-10-11 (updated 2025-10-12)
Status: Accepted

## Context

The initial orchestrator was a single oversized file containing: agent invocation, JSON/tool call parsing, heuristic fallbacks, evidence creation, and logging patterns. This violated SRP, made unit tests brittle, and obscured fallback guarantees. Type + lint complexity increased as one method accumulated timeout handling, parsing branches, and emergency recovery logic.

## Decision

Adopt a modular, staged pipeline split across dedicated modules:

1. Primary Agent Run (timeout bound)
2. Execute Parsed Tool Calls (suggested JSON inside last AI message)
3. Heuristic Generation (derive tool calls from AI content + raw text)
4. Forced Minimal Write (guaranteed baseline evidence + entities + observations)
5. Finalize (evidence linking + relation retry)

Module allocation:
* `agent_orchestrator.py` – Stage orchestration, evidence utilities, logging, relation retry coordination
* `agent_execution.py` – Async agent call (`run_agent_async`), timeout wrapper (`run_agent_with_timeout`), message processing, suggested tool ordering/execution
* `fallback_strategies.py` – Heuristic entity mining, forced minimal write, generated call synthesis
* `agent_prompts.py` – Prompt template constant
* `json_utils.py` / `ToolCallExtractor` – Resilient JSON extraction & normalization

Public thin wrappers maintain test stability (evidence naming, tool invocation, heuristic entity collection) while internals can evolve.

## Alternatives Considered

* Incremental clean-up inside monolith – Rejected: risk of regression remained high; complexity ceiling not reduced.
* Introduce a planning DSL – Over-engineered for current scale; added learning curve.
* Pure heuristic approach (remove LLM) – Loss of adaptive extraction and semantic richness.

## Consequences

Positive:
* Functions stay under targeted length & cognitive complexity thresholds.
* Clear extension seams (add/modify stage vs. expanding a giant method).
* Improved test isolation (each helper/module directly testable).
* Deterministic fallback ladder ensures no silent no-op chunks.

Trade-offs / Neutral:
* Slight overhead from additional module boundaries.
* Documentation must track multiple files (mitigated via this ADR + updated design doc).

## Implementation Notes (Updated)

* `_run_with_timeout` inside orchestrator is now a thin backward-compatibility shim; canonical timeout logic lives in `agent_execution.run_agent_with_timeout`.
* Helper `_collect_capitalized_entities` removed; replaced by `fallback_strategies.collect_capitalized_entities` (exported for reuse/tests).
* Legacy ad-hoc JSON parsing superseded by consolidated `ToolCallExtractor` using `json_utils` (orjson-based) for speed + resilience.
* Tool execution ordering relocated to `agent_execution.order_tool_calls` to preserve expected sequence (observations → entities → other → relations).
* Relation retry dedup logic extracted via `_normalize_relation_for_retry` + `_relation_key` helpers.

## Metrics / Validation

* Unit tests cover timeout wrapper presence (monkeypatch compatibility) and heuristic fallbacks.
* Manual and test logs confirm: stage banners, truncated AI output, clear origin prefixes ("generated", "forced").
* Line count reduction in orchestrator keeps it below monolith guard threshold (guard script external, not part of ADR scope).

## Future Work

* Adaptive timeout heuristics (chunk length, historical agent latency).
* Metrics export (Prometheus) for stage durations, fallback frequency, retry counts.
* Re-enrichment job to revisit minimal chunks with improved models.
* Cross-chunk entity consolidation / alias resolution pipeline.

## Status

Implemented and documented (see updated `docs/agent_orchestrator.md` ).
