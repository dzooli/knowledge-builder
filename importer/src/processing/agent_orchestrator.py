import asyncio
import contextlib
import json
import re
from typing import Any, Dict, List, cast, Optional

from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage, ToolMessage
from loguru import logger

from ..config import Config
from ..utils import TextUtils, ToolCallExtractor
from ..connectors import Neo4jMemoryConnector


# Prompts
PROMPT_TMPL = """You are a professional linguist and a professional journalist in one person. Below the >>TEXTSTART a text will be embedded and I need to extract key information to build a knowledge base based on widely spreaded information. 

You should understand the text, extract entities like:
EXAMPLE ENTITY TYPES TO LOOK FOR (there could be more...):
- Person: individuals, names, roles, titles
- Organization: companies, institutions, agencies, groups
- Location: cities, countries, addresses, places
- Event: meetings, conferences, incidents, announcements
- Concept: technologies, methodologies, ideas, topics
- Method: algorithms, techniques, processes with or without steps
- Tool: tools, tools with features, products, products with features
- Product: software, services, tools, offerings
- Date: specific dates, time periods, deadlines
- Amount: money, quantities, percentages, metrics

Into each entity some key observations  (like use cases, facts, numbers, prices, owners, web pages, email addresses, addresses, etc.) and the entity type and category needs to be also extracted.

You also need to determine the relations between the existing or newly extracted entities.
MORE RELATIONSHIP TYPES (examples):
- works_for, employed_by, ceo_of, founded_by, owns
- located_in, headquartered_in, based_in, operates_in
- announced, acquired, merged_with, partnered_with
- developed, created, released, launched, offers
- attended, spoke_at, organized, participated_in
- related_to, part_of, depends_on, uses, implements
- provides, needs, costs, contains, requires
Good relation example: entity1 -> efficiently handles -> entity2
Bad relation example: entity1 -> handles -> entity2 efficiently; do not mix the entities with the relation types.
Relation type should be lowercase in the final output.

EXTRACTION STRATEGY:
- Start by searching for key entities, then extract systematically
- Look for proper nouns (capitalized words/phrases)
- Identify relationships indicated by verbs and prepositions
- Extract specific dates, amounts, and measurements
- Note roles, titles, and positions mentioned
- Capture business relationships and hierarchies
- Record events, announcements, technologies, products and activities
- Deletion of existing entity is not allowed
- Since you are a knowledge builder, try to connect the new information to the existing
- START ANSWER: Write "JSONSTART" before each JSON object (or object list) output

The required output: 

  You should follow these rules:
  - You just have the following (and ONLY THE FOLLOWING) external tools with the required parameters to manipulate a knowledge graph database:
    - read_graph: query: str ; for searching memories based on a query containing search terms
    - find_memories_by_name: names[string]
    - create_entities: entities[{ name, type, observations }] where observations is a list of strings (can be empty)
    - create_relations: relations[{ source, relationType, target }]
    - add_observations: observations[{ entityName, observations:str[] }] where observations is a list of strings
  - Evidence entity is a normal entity with the following properties and observations: name="Evidence {SOURCE_ID}-{CHUNK_ID}", type="Evidence", observations=["srcId={SOURCE_ID}", "chunk={CHUNK_ID}", "url={SOURCE_URL}"]
  - Create Evidence entity: "Evidence {SOURCE_ID}-{CHUNK_ID}" based on the provided document-id and chunk-id on the first lines of the text
  - Link all newly observed entities to Evidence with "evidence" relations (use create_relations tool)
  - Search for existing entities using the relevant tools (find_memories_by_name, read_graph, search_memories) before creating new ones and update the observations only when found.
  - Create meaningful relationships between entities
  - Add rich observations with context and details
  - Use canonical names (e.g., "John Smith" not "Mr. Smith")
  - Be thorough - extract as much structured knowledge as possible
  - If the new relations contains entities as targets which are not created by you, you must create them with a proper Category and with Type=Placeholder.
  - The required output should be ONLY a json formatted, machine-readable JSON object with an array (without any unformatted or irrelevant repetition in other format) of tool calls in a proper order (I mean: create the evidence entity, add_observations first, then create_entites including the observations, then create_relations) using this schema below (method is always "tools/call", you should fill the THE_TOOL_NAME and THE_REQUIRED_TOOL_PARAMETERS_AS_DEFINED_ABOVE):
    {[{"method": "tools/call", "params": {"tool_name": "<THE_TOOL_NAME>", "arguments": {REQUIRED_TOOL_PARAMETERS_AS_DEFINED_ABOVE}}},...]}

VERY IMPORTANT WORKFLOW RULES, MUST BE ALWAYS FOLLOWED:
  - CREATE THE EVIDENCE ENTITY FIRST
  - CALL add_observations before any create_entities to ensure the UPSERT workflow
  - CALL create_entities FOR THE EXTRACTED ENTITIES ONLY AFTER THE add_observations calls
  - create the relations between the new entitles if any, every new entity MUST have at least 2 relations including the Evidence.
  - when creating a relation ensure both end is created by you or was existing before. You might repeat creation of an entity if needed, this could be handled by the graph database engine but when you define a relation without existing both of source and target entities is not allowed.
  - Duplicated entity is not a problem but missing relation is a serious problem.
  - link new entities to the created evidence entity using relation type 'evidence' and tool: create_relations
  - DO NOT OUTPUT ANYTHING ELSE EXCLUDING THE JSON OUTPUT AND USE ONLY THE PROVIDED TOOLS, THERE IS NO 'link_to_evidence' or 'create_link_to_evidence' or 'link_entity_to_evidence' tools, only 'create_relations' as I mentioned above.
  - ENSURE EVERY NEW ENTITY HAVE AT LEAST TWO RELATIONS INCLUDING THE EVIDENCE
  - ALL NEW ENTITY MUST BE LINKED TO ITS EVIDENCE

>>TEXTSTART
{TEXT}
"""


class AgentOrchestrator:
    """Orchestrates AI agent processing with Neo4j knowledge graph."""

    def __init__(self, neo4j_connector: Neo4jMemoryConnector):
        self.neo4j_connector = neo4j_connector
        # NOTE: The class is intentionally being refactored step-wise to reduce
        # complexity. Newly added private helpers consolidate repeated code
        # paths (evidence entity creation, heuristic entity extraction,
        # observation addition, relation linking and tool invocation). Public
        # behavior and log semantics are preserved for stability.

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

        def _log_unexpected(
            exc: Exception,
        ):  # local tiny helper to DRY exception logging
            logger.warning(
                f"[agent] {log_prefix}tool {name} unexpected error: {type(exc).__name__}: {exc}"
            )

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
            # Parameter or result shape issues
            logger.warning(f"[agent] {log_prefix}tool {name} param/result error: {exc}")
            return f"error: {exc}", False
        except RuntimeError as exc:
            # Underlying connector/runtime error
            logger.warning(f"[agent] {log_prefix}tool {name} runtime error: {exc}")
            return f"error: {exc}", False
        except Exception as exc:  # noqa: BLE001
            _log_unexpected(exc)
            return f"error: {exc}", False

    # Heuristic entity extraction consolidated (used by fallback paths)
    @staticmethod
    def _collect_capitalized_entities(
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
    # Public thin wrappers (exposed for testing & potential external reuse)
    # ------------------------------------------------------------------
    def make_evidence_name(self, source_id: str, chunk_id: str) -> str:
        """Public wrapper for evidence name generation (tests)."""
        return self._make_evidence_name(source_id, chunk_id)

    def build_evidence_entity(
        self, source_id: str, chunk_id: str, source_url: str
    ) -> Dict[str, Any]:
        """Public wrapper for evidence entity construction (tests)."""
        return self._build_evidence_entity(source_id, chunk_id, source_url)

    def collect_capitalized_entities(
        self, text: str, max_scan_words: int = 50, max_entities: int = 3
    ) -> List[str]:
        """Public wrapper for heuristic capitalized entity extraction."""
        return self._collect_capitalized_entities(text, max_scan_words, max_entities)

    def invoke_tool_safe(self, name: str, params: Dict[str, Any]) -> tuple[Any, bool]:
        """Public wrapper for unified tool invocation returning (result, wrote)."""
        return self._invoke_tool(name, params)

    def add_context_observation(
        self, entity_name: str, text: str, prefix: str = "Context"
    ) -> None:
        """Public wrapper to append a context observation to an entity."""
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
    @staticmethod
    def _guess_entity_type(context_text: str) -> str:
        lowered = context_text.lower()
        if any(
            w in lowered
            for w in ["person", "people", "said", "stated", "mr", "ms", "dr"]
        ):
            return "Person"
        if any(w in lowered for w in ["city", "country", "location", "address"]):
            return "Location"
        if any(
            w in lowered for w in ["event", "meeting", "conference", "announcement"]
        ):
            return "Event"
        if any(w in lowered for w in ["company", "organization", "corp", "inc", "ltd"]):
            return "Organization"
        return "Concept"

    def _fallback_entities_from_text(
        self, text: str, limit: int = 3
    ) -> List[Dict[str, Any]]:
        names = self._collect_capitalized_entities(text, max_entities=limit)
        entities: List[Dict[str, Any]] = []
        etype = self._guess_entity_type(text)
        for n in names:
            entities.append({"name": n, "type": etype, "observations": []})
        return entities

    @staticmethod
    def _extract_entities_from_ai_content(ai_content: str) -> List[Dict[str, Any]]:
        """Pattern-based light extraction from AI content to seed fallback tool calls."""
        entities: List[Dict[str, Any]] = []
        patterns_types = [
            (r"(?:person|individual|people?):?\s*([A-Z][a-zA-Z\s]{2,30})", "Person"),
            (
                r"(?:company|organization|corp|inc|ltd):?\s*([A-Z][a-zA-Z\s&]{2,40})",
                "Organization",
            ),
            (
                r"(?:location|city|country|place):?\s*([A-Z][a-zA-Z\s]{2,30})",
                "Location",
            ),
            (r"(?:event|meeting|conference):?\s*([A-Z][a-zA-Z\s]{2,40})", "Event"),
        ]
        seen: set[str] = set()
        for pattern, etype in patterns_types:
            try:
                matches = re.findall(pattern, ai_content, re.IGNORECASE)
            except re.error:
                continue
            for match in matches[:2]:  # limit each type
                name = match.strip()
                if len(name) > 3 and name not in seen:
                    entities.append({"name": name, "type": etype, "observations": []})
                    seen.add(name)
        return entities

    async def run_agent(self, prompt: str) -> Dict[str, Any]:
        """Run AI agent with given prompt."""
        tools = await self.neo4j_connector.ensure_mcp_tools()
        logger.info(f"[agent] loaded {len(tools)} tools: {[t.name for t in tools]}")

        model = ChatOllama(
            model=Config.OLLAMA_MODEL,
            base_url=Config.OLLAMA_URL,
            temperature=0.1,  # Low temperature for more consistent tool usage
            # Remove format="json" as it may confuse the ReAct agent
        )

        # Configure the agent with recursion limits and timeout
        graph = create_react_agent(model, tools)

        # Set recursion and step limits
        config = {
            "recursion_limit": 15,  # Limit recursive calls
            "step_timeout": 60,  # 60 second timeout per step
        }

        state: Any = {"messages": prompt}

        logger.info(
            f"[agent] invoking agent with prompt length: {len(prompt)}, recursion_limit=15"
        )

        try:
            # cast config to Any to satisfy typing since LangChain Runnable accepts mapping-like
            result = await graph.ainvoke(cast(Any, state), config=cast(Any, config))
        except (asyncio.TimeoutError,) as exc:
            logger.error(f"[agent] agent timeout: {exc}")
            return {"messages": []}
        except (ValueError, TypeError) as exc:
            logger.error(f"[agent] agent config/state error: {exc}")
            return {"messages": []}
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[agent] agent execution failed: {exc}")
            return {"messages": []}

        # Debug: log the raw result structure
        if isinstance(result, dict) and "messages" in result:
            messages = result["messages"]
            logger.info(f"[agent] received {len(messages)} messages from agent")
            for i, msg in enumerate(messages[-3:]):  # Log last 3 messages
                msg_type = type(msg).__name__
                content_preview = str(getattr(msg, "content", ""))[:200]
                tool_calls = getattr(msg, "tool_calls", None)
                logger.info(
                    f"[agent] message {i}: {msg_type} - {content_preview} - tool_calls: {len(tool_calls) if tool_calls else 0}"
                )
        else:
            logger.error("[agent] received no messages from agent")

        return result

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

    def _iter_valid_calls(self, ordered_calls: List[Dict[str, Any]]):
        for i, call in enumerate(ordered_calls, start=1):
            name = call.get("name")
            params = call.get("parameters")
            if self._is_valid_call(name, params):
                yield i, cast(str, name), cast(Dict[str, Any], params)

    def _invoke_and_collect(
        self, index: int, name: str, params: Dict[str, Any]
    ) -> tuple[bool, List[str]]:
        try:
            logger.info(
                f"[agent] executing suggested tool #{index}: {name} args={TextUtils.truncate_text(params, Config.LOG_TOOL_PREVIEW_MAX)}"
            )
            _, wrote = self._invoke_tool(name, params, log_prefix=f"suggested#{index} ")
            touched = self._collect_touched_names(name, params)
            return wrote, touched
        except (ValueError, TypeError) as exc:  # noqa: BLE001 (narrowed)
            logger.warning(f"[agent] suggested tool error {name}: {exc}")
            return False, []

    def execute_tool_calls(self, calls: List[Dict[str, Any]]) -> tuple[bool, List[str]]:
        """Execute tool calls in the proper order."""
        if not calls:
            return False, []

        wrote = False
        touched: List[str] = []

        ordered_calls = self._order_tool_calls(calls)

        for idx, name, params in self._iter_valid_calls(ordered_calls):
            success, names = self._invoke_and_collect(idx, name, params)
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
        except Exception as exc:  # noqa: BLE001 (unexpected - keep pipeline running)
            logger.warning(f"[agent] evidence entity unexpected failure: {exc}")

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
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[agent] evidence relation unexpected failure: {exc}")

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
        """Run the async agent with a global timeout from sync context.

        Rationale:
        The previous implementation nested asyncio.wait_for + asyncio.run which is
        unnecessary and can mis-handle cancellation. This version:
          1. Creates the coroutine once.
          2. Uses asyncio.run with an inner wait_for (single event loop creation).
          3. Gracefully handles TimeoutError and generic exceptions.
        Returns a list of messages (may be empty on failure/timeout).
        """

        async def _runner() -> List[Any]:  # local helper to keep scope tight
            try:
                result = await asyncio.wait_for(self.run_agent(prompt), timeout=timeout)
            except asyncio.TimeoutError:
                logger.error(f"[agent] agent execution timed out after {timeout} sec")
                return []
            except (ValueError, TypeError) as exc:  # configuration/state issues
                logger.error(f"[agent] agent run config/state error: {exc}")
                return []
            except Exception as exc:  # noqa: BLE001 (keep broad to protect outer sync path)
                logger.error(f"[agent] agent execution unexpected failure: {exc}")
                return []
            messages = result.get("messages") if isinstance(result, dict) else []
            return messages if isinstance(messages, list) else []

        try:
            return asyncio.run(_runner())
        except RuntimeError as loop_err:
            # In rare cases (already inside running loop) we fall back to creating a new loop.
            logger.warning(
                f"[agent] reuse existing loop fallback triggered: {loop_err}"
            )
            try:
                loop = asyncio.new_event_loop()
                try:
                    asyncio.set_event_loop(loop)
                    return loop.run_until_complete(_runner())
                finally:
                    try:
                        loop.close()
                    except Exception:  # noqa: BLE001
                        pass
            except Exception as exc:  # noqa: BLE001  (secondary loop failure)
                logger.error(f"[agent] secondary loop run failed: {exc}")
                return []
        except Exception as exc:  # noqa: BLE001 (outer unexpected failure)
            logger.error(f"[agent] unexpected failure running agent: {exc}")
            return []

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

        def _normalize_quotes(s: str) -> str:
            # Replace common smart quotes with ASCII equivalents to improve JSON parsing robustness
            return (
                s.replace("“", '"')
                .replace("”", '"')
                .replace("‘", "'")
                .replace("’", "'")
            )

        def _strip_code_fences(s: str) -> str:
            if "```" in s:
                parts = s.split("```")
                # Try to return a plausible JSON block inside fences
                for i in range(0, len(parts) - 1, 2):
                    block = parts[i + 1]
                    if block.lstrip().startswith("[") or block.lstrip().startswith("{"):
                        return block
                # Fallback: remove the backticks and join inner content
                return "".join(parts[i] for i in range(1, len(parts), 2))
            return s

        def _extract_json_text(s: str) -> Optional[str]:
            s = _normalize_quotes(s)
            # If the model outputs "JSONSTART" markers, prefer the last block following it
            marker = "JSONSTART"
            if marker in s:
                s = s.split(marker)[-1]
            s = _strip_code_fences(s).strip()
            # Quick path
            if s.startswith("[") or s.startswith("{"):
                return s
            # Heuristic scan for a JSON array/object substring
            for open_ch, close_ch in (("[", "]"), ("{", "}")):
                start = s.find(open_ch)
                if start != -1:
                    depth = 0
                    in_str = False
                    esc = False
                    for idx in range(start, len(s)):
                        ch = s[idx]
                        if in_str:
                            if esc:
                                esc = False
                            elif ch == "\\":
                                esc = True
                            elif ch == '"':
                                in_str = False
                        else:
                            if ch == '"':
                                in_str = True
                            elif ch == open_ch:
                                depth += 1
                            elif ch == close_ch:
                                depth -= 1
                                if depth == 0:
                                    return s[start : idx + 1]
            return None

        def _parse_calls_from_content(s: str) -> List[Dict[str, Any]]:
            text = _extract_json_text(s)
            if not text:
                return []
            try:
                obj = json.loads(text)
            except (json.JSONDecodeError, TypeError, ValueError):
                return []
            items = obj if isinstance(obj, list) else [obj]
            normalized: List[Dict[str, Any]] = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                # Schema with method/params
                method = item.get("method")
                params = (
                    item.get("params") if isinstance(item.get("params"), dict) else {}
                )
                tool_name = None
                arguments: Any = {}
                if method == "tools/call":
                    tool_name = (
                        params.get("tool_name") if params else item.get("tool_name")
                    )
                    arguments = (
                        (params.get("arguments") if params else None)
                        or item.get("arguments")
                        or {}
                    )
                # Already-normalized schema (legacy)
                if not tool_name and isinstance(item.get("name"), str):
                    tool_name = item.get("name")
                    arguments = (
                        item.get("parameters")
                        if isinstance(item.get("parameters"), dict)
                        else {}
                    )
                if isinstance(tool_name, str) and isinstance(arguments, dict):
                    normalized.append({"name": tool_name, "parameters": arguments})
            return normalized

        # First, parse tool calls using the new "tools/call" schema
        calls = _parse_calls_from_content(raw_content)

        # Fallback to legacy extractor
        if not calls:
            try:
                calls = ToolCallExtractor.extract_tool_calls(raw_content)
            except (ValueError, TypeError):
                calls = []

        if not calls:
            return False, [], []

        exec_wrote, exec_touched = self.execute_tool_calls(calls)
        return exec_wrote, exec_touched, self.extract_relations_from_calls(calls)

    def _force_minimal_write(
        self, source_id: str, chunk_id: str, source_url: str, text: str
    ) -> tuple[bool, List[str]]:
        try:
            logger.warning(
                f"[agent] forcing minimal write with basic entity extraction for doc={source_id} {chunk_id}"
            )

            # Try to extract at least some basic entities from the text
            entities_to_create = []
            created_entity_names = []

            # Create Evidence entity via helper
            evidence_entity = self._make_evidence_name(source_id, chunk_id)
            entities_to_create.append(
                self._build_evidence_entity(source_id, chunk_id, source_url)
            )
            created_entity_names.append(evidence_entity)

            # Simple heuristic extraction via helper
            potential_entities = self._collect_capitalized_entities(text)

            # Create entities for the most promising candidates (up to 3)
            for entity_name in potential_entities[:3]:
                # Guess entity type based on context
                entity_type = "Organization"
                if any(
                    word in text.lower()
                    for word in ["person", "people", "said", "stated", "mr", "ms", "dr"]
                ):
                    entity_type = "Person"
                elif any(
                    word in text.lower()
                    for word in ["city", "country", "location", "address"]
                ):
                    entity_type = "Location"
                elif any(
                    word in text.lower()
                    for word in ["event", "meeting", "conference", "announcement"]
                ):
                    entity_type = "Event"

                entities_to_create.append(
                    {"name": entity_name, "type": entity_type, "observations": []}
                )
                created_entity_names.append(entity_name)

            # If no entities extracted, create a generic document entity
            if len(entities_to_create) == 1:  # Only Evidence entity
                doc_entity = f"Document {source_id}"
                entities_to_create.append(
                    {"name": doc_entity, "type": "Document", "observations": []}
                )
                created_entity_names.append(doc_entity)

            # Step 1: Create all entities
            entities_payload = {"entities": entities_to_create}
            result_create, _ = self._invoke_tool(
                "create_entities", entities_payload, log_prefix="forced "
            )
            logger.info(
                f"[agent] forced create_entities result: {TextUtils.truncate_text(result_create, Config.LOG_TOOL_PREVIEW_MAX)}"
            )

            # Step 2: Add observations to non-evidence entities
            if not str(result_create).lower().startswith("error"):
                for entity_name in created_entity_names:
                    if entity_name != evidence_entity:
                        # Add relevant text snippet as observation
                        obs_text = text if len(text) <= 800 else text[:800]
                        observations_payload = {
                            "observations": [
                                {
                                    "entityName": entity_name,
                                    "observations": [f"Mentioned in: {obs_text}"],
                                }
                            ]
                        }
                        self._invoke_tool(
                            "add_observations",
                            observations_payload,
                            log_prefix="forced ",
                        )
                        logger.debug(f"[agent] added observations to {entity_name}")

                # Step 3: Create evidence relations for all entities
                relations = []
                for entity_name in created_entity_names:
                    if entity_name != evidence_entity:
                        relations.append(
                            {
                                "source": entity_name,
                                "relationType": "evidence",
                                "target": evidence_entity,
                            }
                        )

                if relations:
                    relations_payload = {"relations": relations}
                    result_rel, _ = self._invoke_tool(
                        "create_relations", relations_payload, log_prefix="forced "
                    )
                    logger.info(
                        f"[agent] forced create_relations result: {TextUtils.truncate_text(result_rel, Config.LOG_TOOL_PREVIEW_MAX)}"
                    )

            logger.info(
                f"[agent] forced minimal write completed for doc={source_id} {chunk_id} - created {len(created_entity_names)} entities"
            )
            return True, created_entity_names

        except (ValueError, TypeError) as exc:
            logger.error(
                f"[agent] forced write data error doc={source_id} {chunk_id}: {exc}",
                exc_info=False,
            )
            return False, []
        except RuntimeError as exc:
            logger.error(
                f"[agent] forced write runtime error doc={source_id} {chunk_id}: {exc}",
                exc_info=False,
            )
            return False, []
        except Exception as exc:  # noqa: BLE001
            logger.error(
                f"[agent] forced write unexpected failure doc={source_id} {chunk_id}: {exc}",
                exc_info=True,
            )
            return False, []

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
            except Exception as exc:
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
        try:
            if not messages or not isinstance(messages[-1], AIMessage):
                return False, []
            last_ai_message = messages[-1]
            ai_content = getattr(last_ai_message, "content", "") or ""

            logger.info(
                f"[agent] generating comprehensive tool calls from AI content for doc={source_id} {chunk_id}"
            )

            entities_from_ai = self._extract_entities_from_ai_content(ai_content)
            entities = entities_from_ai or self._fallback_entities_from_text(
                text, limit=3
            )

            evidence_name = self._make_evidence_name(source_id, chunk_id)
            evidence_entity = self._build_evidence_entity(
                source_id, chunk_id, source_url
            )
            full_entity_list = entities + [evidence_entity]

            if not full_entity_list:
                return False, []

            payload = {"entities": full_entity_list}
            result, _ = self._invoke_tool(
                "create_entities", payload, log_prefix="generated "
            )
            logger.info(
                f"[agent] generated create_entities result: {TextUtils.truncate_text(result, Config.LOG_TOOL_PREVIEW_MAX)}"
            )

            created_names = [e["name"] for e in entities] + [evidence_name]
            if not str(result).lower().startswith("error"):
                obs_text = text if len(text) <= 600 else text[:600]
                for name in [n for n in created_names if n != evidence_name]:
                    self._add_context_observation(name, obs_text, prefix="Context")
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
                    self._invoke_tool(
                        "create_relations",
                        {"relations": relations},
                        log_prefix="generated ",
                    )

            logger.info(
                f"[agent] generated comprehensive tool calls completed for doc={source_id} {chunk_id} - created {len(created_names)} entities"
            )
            return True, created_names

        except Exception as exc:
            logger.warning(f"[agent] failed to generate tool calls from content: {exc}")

        return False, []

    # ------------------------------------------------------------------
    # Modular chunk processing stages
    # ------------------------------------------------------------------
    def _stage_primary_prompt(
        self, source_id: str, chunk_id: str, source_url: str, text: str
    ) -> tuple[bool, List[str], List[Dict[str, Any]], List[Any]]:
        self._log_stage("primary", source_id, chunk_id, "starting")
        prompt = self._build_prompt(PROMPT_TMPL, source_id, chunk_id, source_url, text)
        messages = self._run_with_timeout(prompt)
        self._log_last_message_output(messages, source_id, chunk_id)
        wrote, touched, relations_to_retry = self._process_messages(messages)
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
        exec_wrote, exec_touched, exec_rel = self._execute_suggested_calls_from_last_ai(
            messages
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
            self._retry_relations(relations_to_retry)
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
