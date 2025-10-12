import json
import contextlib
from typing import Any, Dict, List, cast, Optional, Callable, Awaitable

from langchain_core.messages import AIMessage, ToolMessage
from loguru import logger

from ..config import Config
from ..utils import TextUtils, ToolCallExtractor
from ..connectors import Neo4jMemoryConnector
from .agent_prompts import PROMPT_TMPL
from .fallback_strategies import (
    collect_capitalized_entities,
    force_minimal_write,
    generate_tool_calls_from_content,
)
from .agent_execution import (
    run_agent_with_timeout,
    process_messages,
    execute_suggested_calls_from_last_ai,
    order_tool_calls,
    iter_valid_calls,
    invoke_and_collect,
    retry_relations,
)


class AgentOrchestrator:
    """Orchestrates AI agent processing with Neo4j knowledge graph."""

    def __init__(self, neo4j_connector: Neo4jMemoryConnector):
        self.neo4j_connector = neo4j_connector

    # ------------------------------------------------------------------
    # Core small utilities (duplication reducers)
    # ------------------------------------------------------------------
    @staticmethod
    def _make_evidence_name(source_id: str, chunk_id: str) -> str:
        return f"Evidence {source_id}-{chunk_id}"

    @classmethod
    def _build_evidence_entity(
        cls, source_id: str, chunk_id: str, source_url: str
    ) -> Dict[str, Any]:
        return {
            "name": cls._make_evidence_name(source_id, chunk_id),
            "type": "Evidence",
            "observations": [
                f"srcId={source_id}",
                f"chunk={chunk_id}",
                f"url={source_url or ''}",
            ],
        }

    @staticmethod
    def _truncate_text_for_obs(text: str, limit: int = 800) -> str:
        return text if len(text) <= limit else text[:limit]

    def _invoke_tool(
        self, name: str, params: Dict[str, Any], log_prefix: str = ""
    ) -> tuple[Any, bool]:
        """Unified tool invocation with consistent logging & success detection."""
        try:
            logger.info(
                f"[agent] {log_prefix}invoke tool {name} args={TextUtils.truncate_text(params, Config.LOG_TOOL_PREVIEW_MAX)}"
            )
            result = self.neo4j_connector.invoke_tool_by_name(name, params)
            logger.info(
                f"[agent] {log_prefix}tool {name} result={TextUtils.truncate_text(result, Config.LOG_TOOL_PREVIEW_MAX)}"
            )
            wrote = self._is_successful_write(name, result)
            return result, wrote
        except (ValueError, TypeError) as exc:
            logger.warning(f"[agent] {log_prefix}tool {name} param/result error: {exc}")
            return f"error: {exc}", False
        except RuntimeError as exc:
            logger.warning(f"[agent] {log_prefix}tool {name} runtime error: {exc}")
            return f"error: {exc}", False

    def _link_entities_to_evidence(self, evidence_name: str, entity_names: List[str]):
        if not entity_names:
            return
        relations = [
            {"source": n, "relationType": "evidence", "target": evidence_name}
            for n in entity_names
            if n != evidence_name
        ]
        if relations:
            self._invoke_tool("create_relations", {"relations": relations})

    def _add_context_observation(self, entity_name: str, text: str, prefix: str):
        obs_text = self._truncate_text_for_obs(text, 800)
        payload = {
            "observations": [
                {"entityName": entity_name, "observations": [f"{prefix}: {obs_text}"]}
            ]
        }
        self._invoke_tool("add_observations", payload)

    # ------------------------------------------------------------------
    # Public thin wrappers
    # ------------------------------------------------------------------
    def make_evidence_name(self, source_id: str, chunk_id: str) -> str:
        return self._make_evidence_name(source_id, chunk_id)

    def build_evidence_entity(
        self, source_id: str, chunk_id: str, source_url: str
    ) -> Dict[str, Any]:
        return self._build_evidence_entity(source_id, chunk_id, source_url)

    def collect_capitalized_entities(
        self, text: str, max_scan_words: int = 50, max_entities: int = 3
    ) -> List[str]:
        return collect_capitalized_entities(text, max_scan_words, max_entities)

    def invoke_tool_safe(self, name: str, params: Dict[str, Any]) -> tuple[Any, bool]:
        return self._invoke_tool(name, params)

    def add_context_observation(
        self, entity_name: str, text: str, prefix: str = "Context"
    ) -> None:
        self._add_context_observation(entity_name, text, prefix)

    # ------------------------------------------------------------------
    # Name extraction helpers (restored after refactor)
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_names_from_add_observations(params: Dict[str, Any]) -> List[str]:
        observations = params.get("observations") or []
        names: List[str] = []
        for observation in observations:
            if not isinstance(observation, dict):
                continue
            entity_name = observation.get("entityName")
            if isinstance(entity_name, dict):
                entity_name = entity_name.get("name")
            if isinstance(entity_name, str):
                names.append(entity_name)
        return names

    @staticmethod
    def _extract_names_from_create_entities(params: Dict[str, Any]) -> List[str]:
        entities = params.get("entities") or []
        names: List[str] = []
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            name = entity.get("name")
            if isinstance(name, str):
                names.append(name)
        return names

    @staticmethod
    def _extract_names_from_create_relations(params: Dict[str, Any]) -> List[str]:
        relations = params.get("relations") or []
        names: List[str] = []
        for relation in relations:
            if not isinstance(relation, dict):
                continue
            source = relation.get("source")
            target = relation.get("target")
            if isinstance(source, dict):
                source = source.get("name")
            if isinstance(target, dict):
                target = target.get("name")
            if isinstance(source, str) and source:
                names.append(source)
            if isinstance(target, str) and target:
                names.append(target)
        return names

    @staticmethod
    def _dedupe_preserve_order(items: List[str]) -> List[str]:
        seen: set[str] = set()
        ordered: List[str] = []
        for x in items:
            if isinstance(x, str) and x and x not in seen:
                seen.add(x)
                ordered.append(x)
        return ordered

    @staticmethod
    def _order_tool_calls(calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        create_entities = [c for c in calls if c.get("name") == "create_entities"]
        add_observations = [c for c in calls if c.get("name") == "add_observations"]
        create_relations = [c for c in calls if c.get("name") == "create_relations"]
        other = [
            c
            for c in calls
            if c.get("name")
            not in {"create_entities", "add_observations", "create_relations"}
        ]
        # Maintain previous ordering policy: add_observations precede create_entities
        return add_observations + create_entities + other + create_relations

    # ------------------------------------------------------------------
    # Fallback extraction helpers
    # ------------------------------------------------------------------
    # Fallback & heuristic helpers now in fallback_strategies module

    # run_agent now provided by agent_execution (async variant not exposed directly here)

    def extract_relations_from_calls(
        self, calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract relations from tool calls."""
        relations: List[Dict[str, Any]] = []
        for call in calls or []:
            try:
                if call.get("name") != "create_relations":
                    continue
                params = call.get("parameters") or {}
                rels = params.get("relations") if isinstance(params, dict) else None
                if isinstance(rels, list):
                    for rel in rels:
                        if (
                            isinstance(rel, dict)
                            and rel.get("source")
                            and rel.get("relationType")
                            and rel.get("target")
                        ):
                            relations.append(
                                {
                                    "source": rel.get("source"),
                                    "relationType": rel.get("relationType"),
                                    "target": rel.get("target"),
                                }
                            )
            except (KeyError, TypeError, AttributeError):
                continue
        return relations

    @staticmethod
    def _is_valid_call(name: Any, params: Any) -> bool:
        return isinstance(name, str) and isinstance(params, dict)

    @staticmethod
    def _is_successful_write(tool_name: str, result: Any) -> bool:
        if tool_name not in {"create_entities", "create_relations", "add_observations"}:
            return False
        return not (isinstance(result, str) and str(result).lower().startswith("error"))

    def _collect_touched_names(
        self, tool_name: str, params: Dict[str, Any]
    ) -> List[str]:
        extractors = {
            "create_entities": self._extract_names_from_create_entities,
            "add_observations": self._extract_names_from_add_observations,
            "create_relations": self._extract_names_from_create_relations,
        }
        extractor = extractors.get(tool_name)
        return extractor(params) if extractor else []

    # (helpers _iter_valid_calls and _invoke_and_collect moved to agent_execution)

    def execute_tool_calls(self, calls: List[Dict[str, Any]]) -> tuple[bool, List[str]]:
        """Execute tool calls in the proper order."""
        if not calls:
            return False, []

        wrote = False
        touched: List[str] = []
        ordered_calls = order_tool_calls(calls)
        for idx, name, params in iter_valid_calls(ordered_calls):
            success, names = invoke_and_collect(self, idx, name, params)
            wrote = wrote or success
            touched.extend(names)
        return wrote, self._dedupe_preserve_order(touched)

    def ensure_evidence_links(
        self, entity_names: List[str], source_id: str, chunk_id: str, source_url: str
    ):
        """Ensure an evidence entity exists and is linked to all touched entities."""
        if not entity_names:
            return

        evidence_name = self._make_evidence_name(source_id, chunk_id)

        try:
            evidence_entity = self._build_evidence_entity(
                source_id, chunk_id, source_url
            )
            self._invoke_tool(
                "create_entities",
                {"entities": [evidence_entity]},
                log_prefix="evidence ",
            )
        except (ValueError, TypeError) as exc:
            logger.warning(f"[agent] evidence entity build error: {exc}")
        except RuntimeError as exc:
            logger.warning(f"[agent] evidence entity runtime error: {exc}")

        sources = [
            name
            for name in entity_names
            if isinstance(name, str) and name and name != evidence_name
        ]
        if sources:
            relations = [
                {"source": name, "relationType": "evidence", "target": evidence_name}
                for name in sources
            ]
            try:
                self._invoke_tool(
                    "create_relations", {"relations": relations}, log_prefix="evidence "
                )
                logger.info(
                    f"[agent] linked {len(relations)} entities to evidence {evidence_name}"
                )
            except (ValueError, TypeError) as exc:
                logger.warning(f"[agent] evidence relation param error: {exc}")
            except RuntimeError as exc:
                logger.warning(f"[agent] evidence relation runtime error: {exc}")

    # --- Helpers extracted to reduce process_chunk complexity ---
    @staticmethod
    def _log_chunk_start_preview(source_id: str, chunk_id: str, text: str):
        logger.info(f"[chunk] start doc={source_id} {chunk_id} len={len(text)}")
        if Config.LOG_CHUNK_FULL:
            logger.info(f"[chunk] text doc={source_id} {chunk_id}:\n{text}")
        else:
            logger.info(
                f"[chunk] preview doc={source_id} {chunk_id}:\n{TextUtils.truncate_text(text, Config.LOG_CHUNK_PREVIEW_MAX)}"
            )

    # ------------------------------------------------------------------
    # Logging helpers (Task 8 - deduplicate patterns)
    # ------------------------------------------------------------------
    @staticmethod
    def _log_stage(stage: str, source_id: str, chunk_id: str, msg: str):
        """Unified stage log format."""
        logger.info(f"[stage:{stage}] doc={source_id} {chunk_id} {msg}")

    @staticmethod
    def _log_stage_warn(stage: str, source_id: str, chunk_id: str, msg: str):
        logger.warning(f"[stage:{stage}] doc={source_id} {chunk_id} {msg}")

    @staticmethod
    def _build_prompt(
        tmpl: str, source_id: str, chunk_id: str, source_url: str, text: str
    ) -> str:
        return (
            tmpl.replace("{SOURCE_ID}", source_id)
            .replace("{CHUNK_ID}", chunk_id)
            .replace("{SOURCE_URL}", source_url or "")
            .replace("{TEXT}", text)
        )

    def _run_with_timeout(self, prompt: str, timeout: float = 300.0) -> List[Any]:
        """Backward-compatible wrapper for tests expecting this symbol.

        If a test monkeypatches an async ``run_agent`` (prompt -> {"messages": [...]})
        we execute it with a timeout and return the messages; otherwise we
        delegate to the shared implementation.
        """
        fn = getattr(self, "run_agent", None)
        if callable(fn):  # only attempt special path if callable
            import inspect
            import asyncio

            if inspect.iscoroutinefunction(fn):

                async def _runner(
                    call_impl: Callable[[str], Awaitable[Any]],
                ) -> List[Any]:  # pragma: no cover - simple
                    try:
                        res = await asyncio.wait_for(call_impl(prompt), timeout=timeout)
                    except asyncio.TimeoutError:
                        return []
                    if isinstance(res, dict):
                        msgs = res.get("messages")
                        if isinstance(msgs, list):
                            return msgs
                    return []

                typed_fn = cast(Callable[[str], Awaitable[Any]], fn)
                try:
                    return asyncio.run(_runner(typed_fn))
                except RuntimeError:  # already inside loop
                    loop = asyncio.new_event_loop()
                    try:
                        asyncio.set_event_loop(loop)
                        return loop.run_until_complete(_runner(typed_fn))
                    finally:
                        with contextlib.suppress(Exception):
                            loop.close()
        return run_agent_with_timeout(self, prompt, timeout=timeout)

    @staticmethod
    def _log_last_message_output(
        messages: List[Any], source_id: str, chunk_id: str, prefix: str = ""
    ):
        if messages:
            last_message = messages[-1]
            content = getattr(last_message, "content", "")
            try:
                content_text = (
                    content
                    if isinstance(content, str)
                    else json.dumps(content, ensure_ascii=False, indent=2)
                )
            except (json.JSONDecodeError, TypeError, ValueError):
                content_text = str(content)
            logger.info(
                f"[agent] {prefix}output doc={source_id} {chunk_id} (full LLM response):\n{content_text}"
            )
        else:
            logger.warning(f"[agent] {prefix}no output doc={source_id} {chunk_id}")

    @staticmethod
    def _log_ai_tool_calls(message: Any, index: int, prefix: str = ""):
        if getattr(message, "tool_calls", None):
            for tool_call in message.tool_calls:
                logger.info(
                    f"[agent] {prefix}ai#{index} tool_call name={tool_call.get('name')} "
                    f"args={TextUtils.truncate_text(tool_call.get('args'), Config.LOG_TOOL_PREVIEW_MAX)}"
                )

    @staticmethod
    def _parse_message_data(raw: Any) -> Any:
        if isinstance(raw, str):
            with contextlib.suppress(Exception):
                return json.loads(raw)
            return None
        return raw if isinstance(raw, (dict, list)) else None

    def _handle_tool_message(
        self, message: Any, index: int, prefix: str = ""
    ) -> tuple[bool, List[str], List[Dict[str, Any]]]:
        logger.info(
            f"[agent] {prefix}tool#{index} name={message.name} result={TextUtils.truncate_text(message.content, Config.LOG_TOOL_PREVIEW_MAX)}"
        )

        wrote = False
        if message.name in {
            "create_entities",
            "create_relations",
            "add_observations",
            "delete_entities",
            "delete_relations",
            "delete_observations",
        }:
            content = (
                (message.content or "")
                if isinstance(message.content, str)
                else json.dumps(message.content)
            )
            if not str(content).lower().startswith("error"):
                wrote = True

        touched: List[str] = []
        relations_to_retry: List[Dict[str, Any]] = []
        data = self._parse_message_data(message.content)

        if message.name == "add_observations" and isinstance(data, list):
            touched += [
                str(d.get("entityName"))
                for d in data
                if isinstance(d, dict)
                and isinstance(d.get("entityName"), str)
                and d.get("entityName") is not None
            ]
        if message.name == "create_entities" and isinstance(data, list):
            touched += [
                str(d.get("name"))
                for d in data
                if isinstance(d, dict)
                and isinstance(d.get("name"), str)
                and d.get("name") is not None
            ]
        if message.name == "create_relations" and isinstance(data, list):
            relations_to_retry.extend(
                relation for relation in data if isinstance(relation, dict)
            )

        return wrote, touched, relations_to_retry

    def _process_messages(
        self, messages: List[Any], prefix: str = ""
    ) -> tuple[bool, List[str], List[Dict[str, Any]]]:
        wrote = False
        touched: List[str] = []
        relations_to_retry: List[Dict[str, Any]] = []
        has_tool_calls = False

        for i, message in enumerate(messages or [], start=1):
            try:
                if isinstance(message, AIMessage):
                    self._log_ai_tool_calls(message, i, prefix)
                    # Check if this AI message has tool calls
                    if getattr(message, "tool_calls", None):
                        has_tool_calls = True
                elif isinstance(message, ToolMessage):
                    has_tool_calls = True
                    w, t, r = self._handle_tool_message(message, i, prefix)
                    wrote = wrote or w
                    touched.extend(t)
                    relations_to_retry.extend(r)
            except (ValueError, TypeError, AttributeError) as exc:
                logger.warning(f"[agent] {prefix}msg#{i} processing error: {exc}")

        # Debug logging
        if not has_tool_calls:
            logger.warning(
                f"[agent] {prefix}no tool calls detected in {len(messages)} messages"
            )
            if messages:
                last_msg = messages[-1]
                if isinstance(last_msg, AIMessage):
                    content = getattr(last_msg, "content", "")
                    try:
                        content_text = (
                            content
                            if isinstance(content, str)
                            else json.dumps(content, ensure_ascii=False, indent=2)
                        )
                    except (json.JSONDecodeError, TypeError, ValueError):
                        content_text = str(content)
                    logger.info(
                        f"[agent] {prefix}last AI message content (full): {content_text}"
                    )

        return wrote, touched, relations_to_retry

    def _execute_suggested_calls_from_last_ai(
        self, messages: List[Any]
    ) -> tuple[bool, List[str], List[Dict[str, Any]]]:
        if not messages or not isinstance(messages[-1], AIMessage):
            return False, [], []
        last_ai = cast(AIMessage, messages[-1])
        raw_content = str(getattr(last_ai, "content", "") or "")

        # Use simplified JSON extraction via ToolCallExtractor (which uses json_utils)
        # NOTE: Previous implementation had multiple nested helpers for quote normalization,
        # code-fence stripping, balanced scanning, and schema normalization. These have been
        # consolidated into json_utils + ToolCallExtractor to reduce maintenance surface and
        # leverage orjson for performance. If model output format drifts, extend json_utils
        # rather than re-introducing ad-hoc parsing here.
        try:
            calls = ToolCallExtractor.extract_tool_calls(raw_content)
        except (ValueError, TypeError):  # model produced invalid structure
            calls = []

        if not calls:
            return False, [], []

        exec_wrote, exec_touched = self.execute_tool_calls(calls)
        return exec_wrote, exec_touched, self.extract_relations_from_calls(calls)

    def _force_minimal_write(
        self, source_id: str, chunk_id: str, source_url: str, text: str
    ) -> tuple[bool, List[str]]:
        return force_minimal_write(self, source_id, chunk_id, source_url, text)

    # New helpers for relation retry dedup/normalization
    def _normalize_relation_for_retry(
        self, relation: Dict[str, Any]
    ) -> Optional[Dict[str, str]]:
        if not isinstance(relation, dict):
            return None
        source: Any = relation.get("source")
        target: Any = relation.get("target")
        relation_type: Any = relation.get("relationType") or relation.get("predicate")
        if isinstance(source, dict):
            source = source.get("name")
        if isinstance(target, dict):
            target = target.get("name")
        if all(isinstance(x, str) and x for x in (source, relation_type, target)):
            return {
                "source": cast(str, source),
                "relationType": cast(str, relation_type),
                "target": cast(str, target),
            }
        return None

    @staticmethod
    def _relation_key(rel: Dict[str, str]) -> tuple[str, str, str]:
        return rel.get("source", ""), rel.get("relationType", ""), rel.get("target", "")

    def _retry_relations(self, relations_to_retry: List[Dict[str, Any]]):
        if not relations_to_retry:
            return
        final_relations: List[Dict[str, str]] = []
        seen: set[tuple[str, str, str]] = set()

        for relation in relations_to_retry:
            normalized = self._normalize_relation_for_retry(relation)
            if not normalized:
                continue
            key = self._relation_key(normalized)
            if key in seen:
                continue
            seen.add(key)
            final_relations.append(normalized)

        if final_relations:
            try:
                self.neo4j_connector.invoke_tool_by_name(
                    "create_relations", {"relations": final_relations}
                )
                logger.info(
                    f"[agent] retried {len(final_relations)} domain relations at end of chunk"
                )
            except RuntimeError as exc:
                logger.warning(f"[agent] retrying relations failed: {exc}")

    def _generate_tool_calls_from_content(
        self,
        messages: List[Any],
        source_id: str,
        chunk_id: str,
        source_url: str,
        text: str,
    ) -> tuple[bool, List[str]]:
        """Generate comprehensive tool calls when the agent fails to use tools properly."""
        return generate_tool_calls_from_content(
            self, messages, source_id, chunk_id, source_url, text
        )

    # ------------------------------------------------------------------
    # Modular chunk processing stages
    # ------------------------------------------------------------------
    def _stage_primary_prompt(
        self, source_id: str, chunk_id: str, source_url: str, text: str
    ) -> tuple[bool, List[str], List[Dict[str, Any]], List[Any]]:
        self._log_stage("primary", source_id, chunk_id, "starting")
        prompt = self._build_prompt(PROMPT_TMPL, source_id, chunk_id, source_url, text)
        messages = run_agent_with_timeout(self, prompt)
        self._log_last_message_output(messages, source_id, chunk_id)
        wrote, touched, relations_to_retry = process_messages(self, messages)
        self._log_stage(
            "primary",
            source_id,
            chunk_id,
            f"result wrote={wrote} touched_entities={len(touched)}",
        )
        return wrote, touched, relations_to_retry, messages

    def _stage_execute_ai_suggestions(
        self,
        messages: List[Any],
        wrote: bool,
        touched: List[str],
        relations_to_retry: List[Dict[str, Any]],
        source_id: str,
        chunk_id: str,
    ) -> tuple[bool, List[str], List[Dict[str, Any]]]:
        if wrote:
            return wrote, touched, relations_to_retry
        self._log_stage("suggested", source_id, chunk_id, "attempt (no writes)")
        exec_wrote, exec_touched, exec_rel = execute_suggested_calls_from_last_ai(
            self, messages
        )
        wrote = wrote or exec_wrote
        if exec_touched:
            touched.extend(exec_touched)
        if exec_rel:
            relations_to_retry.extend(exec_rel)
        self._log_stage(
            "suggested",
            source_id,
            chunk_id,
            f"result wrote={exec_wrote} additional_touched={len(exec_touched)}",
        )
        return wrote, touched, relations_to_retry

    def _stage_generate_calls_from_content(
        self,
        messages: List[Any],
        wrote: bool,
        touched: List[str],
        source_id: str,
        chunk_id: str,
        source_url: str,
        text: str,
    ) -> tuple[bool, List[str]]:
        if wrote or not messages:
            return wrote, touched
        self._log_stage("generate", source_id, chunk_id, "attempt (suggested failed)")
        generated_wrote, generated_touched = self._generate_tool_calls_from_content(
            messages, source_id, chunk_id, source_url, text
        )
        wrote = wrote or generated_wrote
        if generated_touched:
            touched.extend(generated_touched)
        self._log_stage(
            "generate",
            source_id,
            chunk_id,
            f"result wrote={generated_wrote} additional_touched={len(generated_touched)}",
        )
        return wrote, touched

    def _stage_force_minimal(
        self,
        wrote: bool,
        touched: List[str],
        source_id: str,
        chunk_id: str,
        source_url: str,
        text: str,
    ) -> tuple[bool, List[str]]:
        if wrote or not (text and text.strip()):
            return wrote, touched
        self._log_stage_warn(
            "forced", source_id, chunk_id, "attempt (all previous failed)"
        )
        fw_wrote, fw_touched = self._force_minimal_write(
            source_id, chunk_id, source_url, text
        )
        wrote = wrote or fw_wrote
        if fw_touched:
            touched.extend(fw_touched)
        self._log_stage(
            "forced",
            source_id,
            chunk_id,
            f"result wrote={fw_wrote} touched={len(fw_touched)}",
        )
        return wrote, touched

    def _stage_finalize(
        self,
        wrote: bool,
        touched: List[str],
        relations_to_retry: List[Dict[str, Any]],
        source_id: str,
        chunk_id: str,
        source_url: str,
    ) -> None:
        final_touched = self._dedupe_preserve_order(touched)
        if final_touched:
            self._log_stage(
                "finalize",
                source_id,
                chunk_id,
                f"linking evidence for {len(final_touched)} entities",
            )
            self.ensure_evidence_links(final_touched, source_id, chunk_id, source_url)
        else:
            self._log_stage_warn("finalize", source_id, chunk_id, "no entities to link")
        if relations_to_retry:
            self._log_stage(
                "finalize",
                source_id,
                chunk_id,
                f"retrying {len(relations_to_retry)} relations",
            )
            retry_relations(self, relations_to_retry)
        logger.info(
            f"[chunk] COMPLETED doc={source_id} {chunk_id}: wrote={wrote}, total_entities={len(final_touched)}, relations_retried={len(relations_to_retry)}"
        )

    def process_chunk(
        self, source_id: str, chunk_id: str, source_url: str, text: str
    ) -> None:
        """Process a single chunk of text (modular pipeline)."""
        logger.info(f"[chunk] start doc={source_id} {chunk_id} len={len(text)}")
        self._log_chunk_start_preview(source_id, chunk_id, text)
        wrote = False
        touched: List[str] = []
        relations_to_retry: List[Dict[str, Any]] = []
        try:
            wrote, touched, relations_to_retry, messages = self._stage_primary_prompt(
                source_id, chunk_id, source_url, text
            )
            wrote, touched, relations_to_retry = self._stage_execute_ai_suggestions(
                messages, wrote, touched, relations_to_retry, source_id, chunk_id
            )
            wrote, touched = self._stage_generate_calls_from_content(
                messages, wrote, touched, source_id, chunk_id, source_url, text
            )
            wrote, touched = self._stage_force_minimal(
                wrote, touched, source_id, chunk_id, source_url, text
            )
            self._stage_finalize(
                wrote, touched, relations_to_retry, source_id, chunk_id, source_url
            )
        except Exception as exc:
            logger.error(
                f"[chunk] FAILED doc={source_id} {chunk_id}: {exc}", exc_info=True
            )
            with contextlib.suppress(Exception):
                if not wrote and text and text.strip():
                    logger.info(
                        f"[chunk] attempting emergency fallback for failed doc={source_id} {chunk_id}"
                    )
                    self._force_minimal_write(source_id, chunk_id, source_url, text)
            raise
