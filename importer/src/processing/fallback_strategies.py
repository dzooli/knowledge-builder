"""Fallback and heuristic strategies extracted from agent_orchestrator.

Keeping these in a separate module reduces orchestrator size and isolates
heuristics that may evolve independently.
"""

from __future__ import annotations

from typing import Any, Dict, List
import re
from loguru import logger

from ..config import Config
from ..utils import TextUtils

__all__ = [
    "collect_capitalized_entities",
    "guess_entity_type",
    "fallback_entities_from_text",
    "extract_entities_from_ai_content",
    "force_minimal_write",
    "generate_tool_calls_from_content",
]


def collect_capitalized_entities(
    text: str, max_scan_words: int = 50, max_entities: int = 3
) -> List[str]:
    words = text.split()
    entities: List[str] = []
    for i, word in enumerate(words[:max_scan_words]):
        clean = re.sub(r"[^\w\s]", "", word)
        if clean and clean[0].isupper() and len(clean) > 2:
            candidate = clean
            if i + 1 < len(words):
                nxt = re.sub(r"[^\w\s]", "", words[i + 1])
                if nxt and nxt[0].isupper() and len(nxt) > 1:
                    candidate = f"{clean} {nxt}"
            if candidate not in entities and len(candidate) > 3:
                entities.append(candidate)
            if len(entities) >= max_entities:
                break
    return entities


def guess_entity_type(context_text: str) -> str:
    lowered = context_text.lower()
    if any(
        w in lowered for w in ["person", "people", "said", "stated", "mr", "ms", "dr"]
    ):
        return "Person"
    if any(w in lowered for w in ["city", "country", "location", "address"]):
        return "Location"
    if any(w in lowered for w in ["event", "meeting", "conference", "announcement"]):
        return "Event"
    if any(w in lowered for w in ["company", "organization", "corp", "inc", "ltd"]):
        return "Organization"
    return "Concept"


def fallback_entities_from_text(text: str, limit: int = 3) -> List[Dict[str, Any]]:
    names = collect_capitalized_entities(text, max_entities=limit)
    entities: List[Dict[str, Any]] = []
    etype = guess_entity_type(text)
    for n in names:
        entities.append({"name": n, "type": etype, "observations": []})
    return entities


def extract_entities_from_ai_content(ai_content: str) -> List[Dict[str, Any]]:
    entities: List[Dict[str, Any]] = []
    patterns_types = [
        (r"(?:person|individual|people?):?\s*([A-Z][a-zA-Z\s]{2,30})", "Person"),
        (
            r"(?:company|organization|corp|inc|ltd):?\s*([A-Z][a-zA-Z\s&]{2,40})",
            "Organization",
        ),
        (r"(?:location|city|country|place):?\s*([A-Z][a-zA-Z\s]{2,30})", "Location"),
        (r"(?:event|meeting|conference):?\s*([A-Z][a-zA-Z\s]{2,40})", "Event"),
    ]
    seen: set[str] = set()
    for pattern, etype in patterns_types:
        try:
            matches = re.findall(pattern, ai_content, re.IGNORECASE)
        except re.error:
            continue
        for match in matches[:2]:
            name = match.strip()
            if len(name) > 3 and name not in seen:
                entities.append({"name": name, "type": etype, "observations": []})
                seen.add(name)
    return entities


def force_minimal_write(
    orchestrator: Any,
    source_id: str,
    chunk_id: str,
    source_url: str,
    text: str,
) -> tuple[bool, List[str]]:
    """Perform a minimal write via basic heuristics.

    Orchestrator is passed (duck-typed) for invoking `_invoke_tool` and helper
    methods `_make_evidence_name` / `_build_evidence_entity`.
    """
    try:
        logger.warning(
            f"[agent] forcing minimal write with basic entity extraction for doc={source_id} {chunk_id}"
        )
        entities_to_create: List[Dict[str, Any]] = []
        created_entity_names: List[str] = []
        evidence_entity_name = orchestrator.make_evidence_name(source_id, chunk_id)
        entities_to_create.append(
            orchestrator.build_evidence_entity(source_id, chunk_id, source_url)
        )
        created_entity_names.append(evidence_entity_name)

        potential_entities = collect_capitalized_entities(text)
        for entity_name in potential_entities[:3]:
            entity_type = "Organization"
            lower = text.lower()
            if any(
                word in lower
                for word in ["person", "people", "said", "stated", "mr", "ms", "dr"]
            ):
                entity_type = "Person"
            elif any(
                word in lower for word in ["city", "country", "location", "address"]
            ):
                entity_type = "Location"
            elif any(
                word in lower
                for word in ["event", "meeting", "conference", "announcement"]
            ):
                entity_type = "Event"
            entities_to_create.append(
                {"name": entity_name, "type": entity_type, "observations": []}
            )
            created_entity_names.append(entity_name)

        if len(entities_to_create) == 1:
            doc_entity = f"Document {source_id}"
            entities_to_create.append(
                {"name": doc_entity, "type": "Document", "observations": []}
            )
            created_entity_names.append(doc_entity)

        result_create, _ = orchestrator.invoke_tool_safe(
            "create_entities", {"entities": entities_to_create}
        )
        logger.info(
            f"[agent] forced create_entities result: {TextUtils.truncate_text(result_create, Config.LOG_TOOL_PREVIEW_MAX)}"
        )

        if not str(result_create).lower().startswith("error"):
            for entity_name in [
                n for n in created_entity_names if n != evidence_entity_name
            ]:
                obs_text = text if len(text) <= 800 else text[:800]
                orchestrator.invoke_tool_safe(
                    "add_observations",
                    {
                        "observations": [
                            {
                                "entityName": entity_name,
                                "observations": [f"Mentioned in: {obs_text}"],
                            }
                        ]
                    },
                )
            relations = [
                {
                    "source": name,
                    "relationType": "evidence",
                    "target": evidence_entity_name,
                }
                for name in created_entity_names
                if name != evidence_entity_name
            ]
            if relations:
                result_rel, _ = orchestrator.invoke_tool_safe(
                    "create_relations", {"relations": relations}
                )
                logger.info(
                    f"[agent] forced create_relations result: {TextUtils.truncate_text(result_rel, Config.LOG_TOOL_PREVIEW_MAX)}"
                )
        logger.info(
            f"[agent] forced minimal write completed for doc={source_id} {chunk_id} - created {len(created_entity_names)} entities"
        )
        return True, created_entity_names
    except (ValueError, TypeError, RuntimeError) as exc:
        logger.error(
            f"[agent] forced write failure doc={source_id} {chunk_id}: {exc}",
            exc_info=False,
        )
        return False, []


def generate_tool_calls_from_content(
    orchestrator: Any,
    messages: List[Any],
    source_id: str,
    chunk_id: str,
    source_url: str,
    text: str,
) -> tuple[bool, List[str]]:
    try:
        if not messages or not hasattr(messages[-1], "content"):
            return False, []
        last_ai_message = messages[-1]
        ai_content = getattr(last_ai_message, "content", "") or ""
        logger.info(
            f"[agent] generating comprehensive tool calls from AI content for doc={source_id} {chunk_id}"
        )
        entities_from_ai = extract_entities_from_ai_content(ai_content)
        entities = entities_from_ai or fallback_entities_from_text(text, limit=3)
        evidence_name = orchestrator.make_evidence_name(source_id, chunk_id)
        evidence_entity = orchestrator.build_evidence_entity(
            source_id, chunk_id, source_url
        )
        full_entity_list = entities + [evidence_entity]
        if not full_entity_list:
            return False, []
        result, _ = orchestrator.invoke_tool_safe(
            "create_entities", {"entities": full_entity_list}
        )
        logger.info(
            f"[agent] generated create_entities result: {TextUtils.truncate_text(result, Config.LOG_TOOL_PREVIEW_MAX)}"
        )
        created_names = [e["name"] for e in entities] + [evidence_name]
        if not str(result).lower().startswith("error"):
            obs_text = text if len(text) <= 600 else text[:600]
            for name in [n for n in created_names if n != evidence_name]:
                orchestrator.add_context_observation(name, obs_text, prefix="Context")
            relations = [
                {
                    "source": name,
                    "relationType": "evidence",
                    "target": evidence_name,
                }
                for name in created_names
                if name != evidence_name
            ]
            if relations:
                orchestrator.invoke_tool_safe(
                    "create_relations", {"relations": relations}
                )
        logger.info(
            f"[agent] generated comprehensive tool calls completed for doc={source_id} {chunk_id} - created {len(created_names)} entities"
        )
        return True, created_names
    except (ValueError, TypeError, RuntimeError) as exc:
        logger.warning(f"[agent] failed to generate tool calls from content: {exc}")
    return False, []
