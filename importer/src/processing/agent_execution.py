"""Execution and message-processing helpers for AgentOrchestrator.

This module encapsulates async agent invocation, message parsing, tool call
execution ordering, and relation retry logic to keep the main orchestrator
lightweight and below monolith thresholds.
"""

from __future__ import annotations

from typing import Any, Dict, List, Iterable, Tuple, cast, Protocol
import asyncio
import json
import contextlib
from loguru import logger
from langchain_ollama import ChatOllama

try:  # Lazy optional import to tolerate version differences
    from langgraph.prebuilt import create_react_agent  # type: ignore
except ImportError:
    create_react_agent = None  # type: ignore
from langchain_core.messages import AIMessage, ToolMessage

from ..config import Config
from ..utils import TextUtils, ToolCallExtractor

__all__ = [
    "run_agent_async",
    "run_agent_with_timeout",
    "process_messages",
    "execute_suggested_calls_from_last_ai",
    "order_tool_calls",
    "invoke_and_collect",
    "iter_valid_calls",
    "retry_relations",
]


class OrchestratorProto(Protocol):  # minimal protocol for static analysis
    neo4j_connector: Any

    def execute_tool_calls(
        self, calls: List[Dict[str, Any]]
    ) -> tuple[bool, List[str]]: ...  # noqa: D401,E701
    def extract_relations_from_calls(
        self, calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]: ...  # noqa: D401,E701
    # Public wrappers exposed by orchestrator
    def invoke_tool_safe(
        self, name: str, params: Dict[str, Any]
    ) -> tuple[Any, bool]: ...  # noqa: D401,E701
    def collect_capitalized_entities(
        self, text: str, max_scan_words: int = 50, max_entities: int = 3
    ) -> List[str]: ...  # noqa: D401,E701
    def add_context_observation(
        self, entity_name: str, text: str, prefix: str = "Context"
    ) -> None: ...  # noqa: D401,E701
    def make_evidence_name(self, source_id: str, chunk_id: str) -> str: ...  # noqa: D401,E701
    def build_evidence_entity(
        self, source_id: str, chunk_id: str, source_url: str
    ) -> Dict[str, Any]: ...  # noqa: D401,E701


async def run_agent_async(
    orchestrator: OrchestratorProto, prompt: str
) -> Dict[str, Any]:
    tools = await orchestrator.neo4j_connector.ensure_mcp_tools()
    logger.info(f"[agent] loaded {len(tools)} tools: {[t.name for t in tools]}")
    model = ChatOllama(
        model=Config.OLLAMA_MODEL,
        base_url=Config.OLLAMA_URL,
        temperature=0.1,
    )
    if create_react_agent is None:
        logger.error(
            "[agent] create_react_agent unavailable in current langgraph; returning empty messages"
        )
        return {"messages": []}
    graph = create_react_agent(model, tools)
    config = {"recursion_limit": 15, "step_timeout": 60}
    state: Any = {"messages": prompt}
    logger.info(
        f"[agent] invoking agent with prompt length: {len(prompt)}, recursion_limit=15"
    )
    try:
        result = await graph.ainvoke(cast(Any, state), config=cast(Any, config))
    except asyncio.TimeoutError as exc:  # pragma: no cover - network timing
        logger.error(f"[agent] agent timeout: {exc}")
        return {"messages": []}
    except (ValueError, TypeError) as exc:
        logger.error(f"[agent] agent config/state error: {exc}")
        return {"messages": []}
    except (RuntimeError, AttributeError) as exc:
        logger.error(f"[agent] agent execution failed: {exc}")
        return {"messages": []}
    if isinstance(result, dict) and "messages" in result:
        messages = result["messages"]
        logger.info(f"[agent] received {len(messages)} messages from agent")
        for i, msg in enumerate(messages[-3:]):
            msg_type = type(msg).__name__
            content_preview = str(getattr(msg, "content", ""))[:200]
            tool_calls = getattr(msg, "tool_calls", None)
            logger.info(
                f"[agent] message {i}: {msg_type} - {content_preview} - tool_calls: {len(tool_calls) if tool_calls else 0}"
            )
    else:
        logger.error("[agent] received no messages from agent")
    return result


def run_agent_with_timeout(
    orchestrator: OrchestratorProto, prompt: str, timeout: float = 300.0
) -> List[Any]:
    async def _runner() -> List[Any]:
        try:
            result = await asyncio.wait_for(
                run_agent_async(orchestrator, prompt), timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"[agent] agent execution timed out after {timeout} sec")
            return []
        except (ValueError, TypeError) as exc:
            logger.error(f"[agent] agent run config/state error: {exc}")
            return []
        except (RuntimeError, AttributeError) as exc:
            logger.error(f"[agent] agent execution unexpected failure: {exc}")
            return []
        messages = result.get("messages") if isinstance(result, dict) else []
        return messages if isinstance(messages, list) else []

    try:
        return asyncio.run(_runner())
    except RuntimeError as loop_err:  # already running loop
        logger.warning(f"[agent] reuse existing loop fallback triggered: {loop_err}")
        try:
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(_runner())
            finally:
                with contextlib.suppress(Exception):
                    loop.close()
        except (RuntimeError, ValueError) as exc:
            logger.error(f"[agent] secondary loop run failed: {exc}")
            return []
    except ValueError as exc:  # pragma: no cover - unexpected config
        logger.error(f"[agent] unexpected value error running agent: {exc}")
        return []


def _parse_message_data(raw: Any) -> Any:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
    return raw if isinstance(raw, (dict, list)) else None


def _handle_tool_message(
    message: Any, index: int, prefix: str = ""
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
    data = _parse_message_data(message.content)
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


def process_messages(
    orchestrator: OrchestratorProto, messages: List[Any], prefix: str = ""
) -> tuple[bool, List[str], List[Dict[str, Any]]]:
    wrote = False
    touched: List[str] = []
    relations_to_retry: List[Dict[str, Any]] = []
    has_tool_calls = False
    for i, message in enumerate(messages or [], start=1):
        try:
            if isinstance(message, AIMessage):
                if getattr(message, "tool_calls", None):
                    has_tool_calls = True
                    for tool_call in message.tool_calls:
                        logger.info(
                            f"[agent] {prefix}ai#{i} tool_call name={tool_call.get('name')} args={TextUtils.truncate_text(tool_call.get('args'), Config.LOG_TOOL_PREVIEW_MAX)}"
                        )
            elif isinstance(message, ToolMessage):
                has_tool_calls = True
                w, t, r = _handle_tool_message(message, i, prefix)
                wrote = wrote or w
                touched.extend(t)
                relations_to_retry.extend(r)
        except (ValueError, TypeError, AttributeError) as exc:
            logger.warning(f"[agent] {prefix}msg#{i} processing error: {exc}")
    if not has_tool_calls:
        logger.warning(
            f"[agent] {prefix}no tool calls detected in {len(messages)} messages (orchestrator={type(orchestrator).__name__})"
        )
    return wrote, touched, relations_to_retry


def execute_suggested_calls_from_last_ai(
    orchestrator: OrchestratorProto, messages: List[Any]
) -> tuple[bool, List[str], List[Dict[str, Any]]]:
    if not messages or not isinstance(messages[-1], AIMessage):
        return False, [], []
    last_ai = cast(AIMessage, messages[-1])
    raw_content = str(getattr(last_ai, "content", "") or "")
    try:
        calls = ToolCallExtractor.extract_tool_calls(raw_content)
    except (ValueError, TypeError):
        calls = []
    if not calls:
        return False, [], []
    exec_wrote, exec_touched = orchestrator.execute_tool_calls(calls)
    return exec_wrote, exec_touched, orchestrator.extract_relations_from_calls(calls)


def order_tool_calls(calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    create_entities = [c for c in calls if c.get("name") == "create_entities"]
    add_observations = [c for c in calls if c.get("name") == "add_observations"]
    create_relations = [c for c in calls if c.get("name") == "create_relations"]
    other = [
        c
        for c in calls
        if c.get("name")
        not in {"create_entities", "add_observations", "create_relations"}
    ]
    return add_observations + create_entities + other + create_relations


def iter_valid_calls(
    ordered_calls: List[Dict[str, Any]],
) -> Iterable[Tuple[int, str, Dict[str, Any]]]:
    for i, call in enumerate(ordered_calls, start=1):
        name = call.get("name")
        params = call.get("parameters")
        if isinstance(name, str) and isinstance(params, dict):
            yield i, cast(str, name), cast(Dict[str, Any], params)


def invoke_and_collect(
    orchestrator: OrchestratorProto, index: int, name: str, params: Dict[str, Any]
) -> tuple[bool, List[str]]:
    logger.info(
        f"[agent] executing suggested tool #{index}: {name} args={TextUtils.truncate_text(params, Config.LOG_TOOL_PREVIEW_MAX)}"
    )
    result, wrote = orchestrator.invoke_tool_safe(name, params)
    if isinstance(result, str) and result.lower().startswith("error"):
        return False, []
    return wrote, []


def retry_relations(
    orchestrator: OrchestratorProto, relations_to_retry: List[Dict[str, Any]]
):
    if not relations_to_retry:
        return
    final_relations: List[Dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for relation in relations_to_retry:
        if not isinstance(relation, dict):
            continue
        source = relation.get("source")
        rel_type = relation.get("relationType") or relation.get("predicate")
        target = relation.get("target")
        if not (
            isinstance(source, str)
            and isinstance(rel_type, str)
            and isinstance(target, str)
            and source
            and rel_type
            and target
        ):
            continue
        key = (source, rel_type, target)
        if key in seen:
            continue
        seen.add(key)
        final_relations.append(
            {"source": source, "relationType": rel_type, "target": target}
        )
    if final_relations:
        try:
            orchestrator.neo4j_connector.invoke_tool_by_name(
                "create_relations", {"relations": final_relations}
            )
            logger.info(
                f"[agent] retried {len(final_relations)} domain relations at end of chunk"
            )
        except (RuntimeError, ValueError, TypeError) as exc:
            logger.warning(f"[agent] retrying relations failed: {exc}")
