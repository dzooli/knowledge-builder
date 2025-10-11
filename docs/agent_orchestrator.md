# Agent Orchestrator Design

## Overview

The `agent_orchestrator.py` module drives AI-guided structured ingestion of text *chunks* into the Neo4j knowledge graph.
It coordinates: (1) primary AI reasoning using a LangChain ReAct agent, (2) parsing & execution of tool calls emitted by the LLM, and (3) resilient multi‑layer fallbacks that guarantee each chunk leaves a minimal semantic footprint (Evidence + Entities + Relations / Observations) even when the LLM under‑performs.

Core goals:
* Deterministic, bounded runtime per chunk (timeouts + staged fallbacks)
* High observability (structured stage logs + truncated raw LLM output)
* Idempotent, consistently named Evidence nodes enabling traceability
* Progressive enhancement: best‑effort AI → heuristic extraction → forced minimal write
* Testability through public light‑weight wrapper methods around internal helpers

## Processing Stages

Chunk ingestion is decomposed into clearly logged stages (all prefixed with a stage banner through `_log_stage` ). These are invoked by `process_chunk` :

1. Primary Prompt & Agent Setup (`_stage_primary_prompt`)
   - Prepares prompt, initializes tools, and runs the async ReAct agent via `_run_with_timeout` .
   - Captures raw output (truncated via config) for logging transparency.
2. Execute AI Suggestions (`_stage_execute_ai_suggestions`)
   - Parses tool call JSON blocks using `ToolCallExtractor` .
   - Normalizes & executes tool calls through centralized `_invoke_tool` .
   - Establishes an Evidence node and links referenced entities.
3. Generate Calls From Content (`_stage_generate_calls_from_content`)
   - Heuristically extracts candidate entities from chunk text when AI emits too few / no calls.
   - Builds synthetic tool calls (create_entities, add_observations, create_relations) to enrich graph.
4. Force Minimal (`_stage_force_minimal`)
   - Last resort: ensures at least one Evidence node, a generic Topic/Document entity, and a textual observation.
5. Finalize (`_stage_finalize`)
   - Performs any final linking or cleanup; returns list of executed calls (& maybe summaries).

Each stage decides whether to proceed to the next based on sufficiency of already executed semantic operations.

## Timeout Execution Model

`_run_with_timeout` encapsulates async agent execution:
* Uses `asyncio.run` with an inner `asyncio.wait_for` enforcing a configurable timeout.
* On timeout returns an empty action list to trigger heuristic fallbacks instead of raising.
* Falls back to creating a new loop if called inside an existing event loop (robust for embedding contexts).

## Helper Abstractions

To reduce duplication and simplify testing, the following helpers were introduced:
* `_make_evidence_name(chunk_hash: str) -> str` – Stable per-chunk evidence naming.
* `_build_evidence_entity(evidence_name: str, source_id: str) -> dict` – Standardized Evidence entity payload.
* `_invoke_tool(tool_name: str, params: dict) -> dict` – Central tool execution + success/exception logging.
* `_collect_capitalized_entities(text: str) -> set[str]` – Lightweight heuristic entity candidate extractor.
* `_link_entities_to_evidence(evidence_name, entities)` – Creates relations between Evidence and touched entities.
* `_add_context_observation(evidence_name, text)` – Adds a truncated textual observation node linked to Evidence.

Public wrappers ( `make_evidence_name` , `build_evidence_entity` , `collect_capitalized_entities` , `invoke_tool_safe` , `add_context_observation` ) expose a thin layer for tests without granting fragile access to Python privates.

## Fallback Strategy

Ordered resilience ladder:
1. AI Structured Calls – Ideal path (rich entity + relation graph updates)
2. Heuristic Augmentation – Supplements sparse AI output by mining proper‑case terms & constructing tool calls
3. Forced Minimal Write – Guarantees baseline graph trace: Evidence + GenericTopic + ObservedText

This ensures *every* chunk yields:
* Traceability: Evidence node named deterministically
* At least one semantic anchor (entity or observation)
* A place for later enrichment (Evidence can be revisited by future reprocessing)

### Fallback Ladder Diagram

```mermaid
flowchart LR
   A[Start Chunk] --> B{Agent Output?}
   B -->|Rich (entities & relations)| C[Execute Parsed Tool Calls]
   C --> H[Finalize]
   B -->|Empty / Sparse| D[Heuristic Augmentation]
   D --> E{Entities Found?}
   E -->|Yes| F[Create Entities / Relations / Observations]
   F --> H
   E -->|No| G[Forced Minimal Write]
   G --> H
   H --> I[Return Executed Call List]
```

## Logging & Observability

* Stage banners via `_log_stage` / `_log_stage_warn` reduce repeated formatting.
* Raw LLM output is logged (truncated to configurable `LOG_LLM_OUTPUT_MAX`).
* All tool invocations pass through `_invoke_tool` which logs: intent, truncated params, success/failure, and exception types.
* Unexpected exceptions are funneled to `_log_unexpected` with contextual data + class name.

## Exception Handling Policy

* Narrow catches for predictable issues (`ValueError`,  `TypeError`,  `json.JSONDecodeError`,  `asyncio.TimeoutError`).
* Broad `Exception` only at outermost safety boundaries to keep pipeline moving—each annotated with rationale and `# noqa` to silence linters intentionally.
* Failures in a single tool call do not abort the entire chunk; fallbacks still proceed.

## Testability Enhancements

Unit & runtime tests exercise:
* Evidence naming & entity construction invariants.
* Heuristic entity extraction logic.
* Safe tool invocation wrapper behavior (success & simulated failure).
* Timeout path returning empty list and triggering fallback.
* Forced minimal fallback when AI output is absent/insufficient.

## Data & Naming Conventions

* Evidence nodes: `Evidence::<chunk_hash_prefix>` (implementation detail may include additional context like doc id).
* Relation verbs kept active voice per Neo4j Memory MCP expectations.
* Observations truncated to config-defined maximum to limit graph bloat.

## Extending the Orchestrator

When adding a new semantic operation:
1. Implement a new tool in the Neo4j Memory MCP or map an existing one.
2. Add a helper if repeated logic emerges (respect SRP & cognitive complexity limits).
3. Invoke through `_invoke_tool` for consistent logging.
4. Add a stage only if it represents a qualitatively new decision layer; otherwise integrate into an existing stage.
5. Add/adjust tests (helpers + runtime) ensuring fallback ladder integrity.

## Performance Considerations

* Minimizes redundant graph writes by early sufficiency checks (skip heuristic/forced stages if AI output is rich).
* Heuristic extraction uses simple regex/word boundary logic—intentionally light weight vs. re-parsing via LLM.
* Centralized truncation prevents log bloat and memory spikes.

## Future Improvements

* Add semantic dedup across documents (cross-chunk entity consolidation heuristics).
* Introduce adaptive timeout (shorter for small chunks, longer for large or relation-heavy suggestions).
* Add metrics exporter (Prometheus) for stage duration & fallback frequency.
* Implement re-enrichment job that revisits minimal Evidence nodes when a more capable model is available.

## Summary

The refactor transforms a monolithic, harder-to-test orchestration flow into a staged, observable, resilient pipeline with explicit extension and fallback seams. This unlocks safer iteration, richer diagnostics, and predictable behavior under degraded AI performance.
