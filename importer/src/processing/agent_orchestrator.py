import asyncio
import contextlib
import json
from typing import Any, Dict, List, cast, Optional

from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage, ToolMessage
from loguru import logger

from config import Config
from utils import TextUtils, ToolCallExtractor
from connectors import Neo4jMemoryConnector


# Prompts
PROMPT_TMPL = """You are a knowledge graph builder. Extract entities, relationships, and facts from the text to build a comprehensive knowledge graph.

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

WORKFLOW:
1. SEARCH FIRST: Use search_memories or find_memories_by_name to check if entities already exist
2. EXTRACT ENTITIES: Identify all entities from the text (aim for 5-15 entities per chunk)
3. CREATE NEW ENTITIES: Only create entities that don't already exist
4. ADD OBSERVATIONS: Add detailed facts, attributes, and context to entities
5. FIND RELATIONSHIPS: Identify how entities relate to each other
6. CREATE RELATIONS: Connect entities with meaningful relationship types
7. CREATE EVIDENCE: Create "Evidence {SOURCE_ID}-{CHUNK_ID}" and link all newly created entities to it
8. FINISH: Write "DONE" when all knowledge is extracted and stored

RELATIONSHIP TYPES (examples):
- works_for, employed_by, ceo_of, founded_by, owns
- located_in, headquartered_in, based_in, operates_in
- announced, acquired, merged_with, partnered_with
- developed, created, released, launched, offers
- attended, spoke_at, organized, participated_in
- related_to, part_of, depends_on, uses, implements

EXTRACTION STRATEGY:
- Look for proper nouns (capitalized words/phrases)
- Identify relationships indicated by verbs and prepositions
- Extract specific dates, amounts, and measurements
- Note roles, titles, and positions mentioned
- Capture business relationships and hierarchies
- Record events, announcements, technologies, products and activities

IMPORTANT:
- Search for existing entities before creating new ones
- Create meaningful relationships between entities
- Add rich observations with context and details
- Use canonical names (e.g., "John Smith" not "Mr. Smith")
- Be thorough - extract as much structured knowledge as possible

TEXT:
{TEXT}

Begin by searching for any key entities that might already exist, then systematically extract all entities and relationships from the text."""

# PROMPT_TMPL_FALLBACK = """Extract structured knowledge from this text for a knowledge graph. From TEXT below, you MUST persist knowledge into the Neo4j memory using the tools.
#
# IMPORTANT: You are BUILDING knowledge, not cleaning up. DO NOT delete anything you create.
#
# WHAT TO EXTRACT:
# - People, organizations, locations, dates, amounts
# - Events, products, concepts, technologies
# - Relationships between entities
# - Facts and attributes about entities
#
# PROCESS:
# 1. Search for existing entities using search_memories
# 2. Create entities for people, companies, places, etc.
# 3. Add observations with facts about each entity
# 4. Create relationships showing how entities connect
# 5. Create Evidence entity: "Evidence {SOURCE_ID}-{CHUNK_ID}"
# 6. Link all entities to Evidence with "evidence" relations
#
# Tool schemas:
# - create_entities: entities[{ name, type, observations }] where observations is a list of strings (can be empty)
# - create_relations: relations[{ source, relationType, target }]
# - add_observations: observations[{ entityName, observations }] where observations is a list of strings
# - Evidence entity: name="Evidence {SOURCE_ID}-{CHUNK_ID}", type="Evidence", observations=["srcId={SOURCE_ID}", "chunk={CHUNK_ID}", "url={SOURCE_URL}"]
#
# AIM FOR DEPTH: Extract 5+ entities and their relationships, not just one generic document node.
#
# TEXT:
# {TEXT}
#
# Start by searching for key entities, then extract systematically."""


class AgentOrchestrator:
    """Orchestrates AI agent processing with Neo4j knowledge graph."""

    def __init__(self, neo4j_connector: Neo4jMemoryConnector):
        self.neo4j_connector = neo4j_connector

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
        graph = create_react_agent(
            model, 
            tools,
            state_modifier="You are a knowledge extraction agent. Use tools to extract information and store it in Neo4j. When you've completed the task, stop immediately."
        )

        # Set recursion and step limits
        config = {
            "recursion_limit": 15,  # Limit recursive calls
            "step_timeout": 60,     # 60 second timeout per step
        }

        state: Any = {"messages": prompt}

        logger.info(f"[agent] invoking agent with prompt length: {len(prompt)}, recursion_limit=15")

        try:
            result = await graph.ainvoke(cast(Any, state), config=config)
        except Exception as exc:
            logger.error(f"[agent] agent execution failed: {exc}")
            # Return a minimal result structure to avoid breaking downstream code
            return {"messages": []}

        # Debug: log the raw result structure
        if isinstance(result, dict) and "messages" in result:
            messages = result["messages"]
            logger.info(f"[agent] received {len(messages)} messages from agent")
            for i, msg in enumerate(messages[-3:]):  # Log last 3 messages
                msg_type = type(msg).__name__
                content_preview = str(getattr(msg, 'content', ''))[:200]
                tool_calls = getattr(msg, 'tool_calls', None)
                logger.info(f"[agent] message {i}: {msg_type} - {content_preview} - tool_calls: {len(tool_calls) if tool_calls else 0}")

        return result

    def extract_relations_from_calls(self, calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract relations from tool calls."""
        relations = []
        for call in calls or []:
            try:
                if call.get("name") != "create_relations":
                    continue
                params = call.get("parameters") or {}
                items = params.get("relations") or []
                relations.extend(r for r in items if isinstance(r, dict))
            except Exception:
                continue
        return relations

    # --- Helpers to reduce complexity in entity extraction ---
    @staticmethod
    def _extract_names_from_add_observations(params: Dict[str, Any]) -> List[str]:
        observations = params.get("observations") or []
        names: List[str] = []
        for observation in observations:
            entity_name = observation.get("entityName") if isinstance(observation, dict) else None
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
            entity_name = entity.get("name") if isinstance(entity, dict) else None
            if isinstance(entity_name, str):
                names.append(entity_name)
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
            if isinstance(source, str) and source and isinstance(target, str) and target:
                names.extend([source, target])
        return names

    @staticmethod
    def _dedupe_preserve_order(items: List[str]) -> List[str]:
        seen: set[str] = set()
        return [x for x in items if isinstance(x, str) and x and (x not in seen and not seen.add(x))]

    def extract_entities_from_calls(self, calls: List[Dict[str, Any]]) -> List[str]:
        """Extract entity names from tool calls with reduced complexity."""
        names: List[str] = []
        for call in calls or []:
            try:
                name = call.get("name")
                params = call.get("parameters") or {}
                if name == "add_observations":
                    names.extend(self._extract_names_from_add_observations(params))
                elif name == "create_entities":
                    names.extend(self._extract_names_from_create_entities(params))
                elif name == "create_relations":
                    names.extend(self._extract_names_from_create_relations(params))
            except Exception:
                continue
        return self._dedupe_preserve_order(names)

    # --- New helpers to lower execute_tool_calls cognitive complexity ---
    @staticmethod
    def _order_tool_calls(calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        create_entities = [c for c in calls if c.get("name") == "create_entities"]
        add_observations = [c for c in calls if c.get("name") == "add_observations"]
        create_relations = [c for c in calls if c.get("name") == "create_relations"]
        other = [c for c in calls if c.get("name") not in {"create_entities", "add_observations", "create_relations"}]
        return create_entities + add_observations + other + create_relations

    @staticmethod
    def _is_valid_call(name: Any, params: Any) -> bool:
        return isinstance(name, str) and isinstance(params, dict)

    @staticmethod
    def _is_successful_write(tool_name: str, result: Any) -> bool:
        if tool_name not in {"create_entities", "create_relations", "add_observations"}:
            return False
        return not (isinstance(result, str) and str(result).lower().startswith("error"))

    def _collect_touched_names(self, tool_name: str, params: Dict[str, Any]) -> List[str]:
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

    def _invoke_and_collect(self, index: int, name: str, params: Dict[str, Any]) -> tuple[bool, List[str]]:
        try:
            logger.info(f"[agent] executing suggested tool #{index}: {name} args={TextUtils.truncate_text(params, Config.LOG_TOOL_PREVIEW_MAX)}")
            result = self.neo4j_connector.invoke_tool_by_name(name, params)
            logger.info(f"[agent] suggested tool result #{index} {name}: {TextUtils.truncate_text(result, Config.LOG_TOOL_PREVIEW_MAX)}")
            wrote = self._is_successful_write(name, result)
            touched = self._collect_touched_names(name, params)
            return wrote, touched
        except Exception as exc:
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

    def ensure_evidence_links(self, entity_names: List[str], source_id: str, chunk_id: str, source_url: str):
        """Ensure an evidence entity exists and is linked to all touched entities."""
        if not entity_names:
            return

        evidence_name = f"Evidence {source_id}-{chunk_id}"

        try:
            # Upsert Evidence entity
            evidence_payload = {"entities": [{
                "name": evidence_name,
                "type": "Evidence",
                "observations": [f"srcId={source_id}", f"chunk={chunk_id}", f"url={source_url or ''}"]
            }]}
            self.neo4j_connector.invoke_tool_by_name("create_entities", evidence_payload)
        except Exception as exc:
            logger.warning(f"[agent] evidence entity upsert failed: {exc}")

        # Create evidence relations from all entities to the evidence
        sources = [name for name in entity_names if isinstance(name, str) and name and name != evidence_name]
        if relations := [
            {"source": name, "relationType": "evidence", "target": evidence_name}
            for name in sources
        ]:
            try:
                self.neo4j_connector.invoke_tool_by_name("create_relations", {"relations": relations})
                logger.info(f"[agent] linked {len(relations)} entities to evidence {evidence_name}")
            except Exception as exc:
                logger.warning(f"[agent] evidence relation creation failed: {exc}")

    # --- Helpers extracted to reduce process_chunk complexity ---
    @staticmethod
    def _log_chunk_start_preview(source_id: str, chunk_id: str, text: str):
        logger.info(f"[chunk] start doc={source_id} {chunk_id} len={len(text)}")
        if Config.LOG_CHUNK_FULL:
            logger.info(f"[chunk] text doc={source_id} {chunk_id}:\n{text}")
        else:
            logger.info(f"[chunk] preview doc={source_id} {chunk_id}:\n{TextUtils.truncate_text(text, Config.LOG_CHUNK_PREVIEW_MAX)}")

    @staticmethod
    def _build_prompt(tmpl: str, source_id: str, chunk_id: str, source_url: str, text: str) -> str:
        return (tmpl
                .replace("{SOURCE_ID}", source_id)
                .replace("{CHUNK_ID}", chunk_id)
                .replace("{SOURCE_URL}", source_url or "")
                .replace("{TEXT}", text))

    def _run_agent_sync(self, prompt: str) -> List[Any]:
        try:
            # Set a timeout for the entire agent run
            result = asyncio.wait_for(self.run_agent(prompt), timeout=300.0)  # 2 minute timeout
            result = asyncio.run(result)
            return result.get("messages") if isinstance(result, dict) else []
        except asyncio.TimeoutError:
            logger.error("[agent] agent execution timed out after 5 minutes")
            return []
        except Exception as exc:
            logger.error(f"[agent] agent execution failed: {exc}")
            return []

    @staticmethod
    def _log_last_message_output(messages: List[Any], source_id: str, chunk_id: str, prefix: str = ""):
        if messages:
            last_message = messages[-1]
            logger.info(f"[agent] {prefix}output doc={source_id} {chunk_id}:\n{TextUtils.truncate_text(getattr(last_message, 'content', ''), Config.LOG_LLM_OUTPUT_MAX)}")

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

    def _handle_tool_message(self, message: Any, index: int, prefix: str = "") -> tuple[bool, List[str], List[Dict[str, Any]]]:
        logger.info(f"[agent] {prefix}tool#{index} name={message.name} result={TextUtils.truncate_text(message.content, Config.LOG_TOOL_PREVIEW_MAX)}")

        wrote = False
        if message.name in {"create_entities", "create_relations", "add_observations", "delete_entities", "delete_relations", "delete_observations"}:
            content = (message.content or "") if isinstance(message.content, str) else json.dumps(message.content)
            if not str(content).lower().startswith("error"):
                wrote = True

        touched: List[str] = []
        relations_to_retry: List[Dict[str, Any]] = []
        data = self._parse_message_data(message.content)

        if message.name == "add_observations" and isinstance(data, list):
            touched += [d.get("entityName") for d in data if isinstance(d, dict) and isinstance(d.get("entityName"), str)]
        if message.name == "create_entities" and isinstance(data, list):
            touched += [d.get("name") for d in data if isinstance(d, dict) and isinstance(d.get("name"), str)]
        if message.name == "create_relations" and isinstance(data, list):
            relations_to_retry.extend(relation for relation in data if isinstance(relation, dict))

        return wrote, touched, relations_to_retry

    def _process_messages(self, messages: List[Any], prefix: str = "") -> tuple[bool, List[str], List[Dict[str, Any]]]:
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
            except Exception as exc:
                logger.warning(f"[agent] {prefix}msg#{i} processing error: {exc}")

        # Debug logging
        if not has_tool_calls:
            logger.warning(f"[agent] {prefix}no tool calls detected in {len(messages)} messages")
            if messages:
                last_msg = messages[-1]
                if isinstance(last_msg, AIMessage):
                    content = getattr(last_msg, 'content', '')
                    logger.info(f"[agent] {prefix}last AI message content: {TextUtils.truncate_text(content, 500)}")

        return wrote, touched, relations_to_retry

    def _execute_suggested_calls_from_last_ai(self, messages: List[Any]) -> tuple[bool, List[str], List[Dict[str, Any]]]:
        if not messages or not isinstance(messages[-1], AIMessage):
            return False, [], []
        last_ai = cast(AIMessage, messages[-1])
        calls = ToolCallExtractor.extract_tool_calls(getattr(last_ai, "content", ""))
        if not calls:
            return False, [], []
        exec_wrote, exec_touched = self.execute_tool_calls(calls)
        return exec_wrote, exec_touched, self.extract_relations_from_calls(calls)

    def _force_minimal_write(self, source_id: str, chunk_id: str, source_url: str, text: str) -> tuple[bool, List[str]]:
        try:
            logger.warning(f"[agent] forcing minimal write with basic entity extraction for doc={source_id} {chunk_id}")

            # Try to extract at least some basic entities from the text
            entities_to_create = []
            created_entity_names = []

            # Create Evidence entity
            evidence_entity = f"Evidence {source_id}-{chunk_id}"
            entities_to_create.append({
                "name": evidence_entity, 
                "type": "Evidence", 
                "observations": [f"srcId={source_id}", f"chunk={chunk_id}", f"url={source_url or ''}"]
            })
            created_entity_names.append(evidence_entity)

            # Simple heuristic extraction of potential entities
            import re
            words = text.split()

            # Look for capitalized words that might be proper nouns
            potential_entities = []
            for i, word in enumerate(words[:50]):  # Only check first 50 words to avoid too much processing
                # Remove punctuation and check if it starts with capital
                clean_word = re.sub(r'[^\w\s]', '', word)
                if clean_word and clean_word[0].isupper() and len(clean_word) > 2:
                    # Look ahead for potential multi-word entities
                    entity_name = clean_word
                    if i + 1 < len(words):
                        next_word = re.sub(r'[^\w\s]', '', words[i + 1])
                        if next_word and next_word[0].isupper() and len(next_word) > 1:
                            entity_name = f"{clean_word} {next_word}"

                    if entity_name not in potential_entities and len(entity_name) > 3:
                        potential_entities.append(entity_name)

            # Create entities for the most promising candidates (up to 3)
            for entity_name in potential_entities[:3]:
                # Guess entity type based on context
                entity_type = "Organization"
                if any(word in text.lower() for word in ["person", "people", "said", "stated", "mr", "ms", "dr"]):
                    entity_type = "Person"
                elif any(word in text.lower() for word in ["city", "country", "location", "address"]):
                    entity_type = "Location"
                elif any(word in text.lower() for word in ["event", "meeting", "conference", "announcement"]):
                    entity_type = "Event"

                entities_to_create.append({
                    "name": entity_name,
                    "type": entity_type, 
                    "observations": []
                })
                created_entity_names.append(entity_name)

            # If no entities extracted, create a generic document entity
            if len(entities_to_create) == 1:  # Only Evidence entity
                doc_entity = f"Document {source_id}"
                entities_to_create.append({
                    "name": doc_entity,
                    "type": "Document",
                    "observations": []
                })
                created_entity_names.append(doc_entity)

            # Step 1: Create all entities
            entities_payload = {"entities": entities_to_create}
            result_create = self.neo4j_connector.invoke_tool_by_name("create_entities", entities_payload)
            logger.info(f"[agent] forced create_entities result: {TextUtils.truncate_text(result_create, Config.LOG_TOOL_PREVIEW_MAX)}")

            # Step 2: Add observations to non-evidence entities
            if not str(result_create).lower().startswith("error"):
                for entity_name in created_entity_names:
                    if entity_name != evidence_entity:
                        # Add relevant text snippet as observation
                        obs_text = text if len(text) <= 800 else text[:800]
                        observations_payload = {"observations": [{"entityName": entity_name, "observations": [f"Mentioned in: {obs_text}"]}]}
                        result_obs = self.neo4j_connector.invoke_tool_by_name("add_observations", observations_payload)
                        logger.debug(f"[agent] added observations to {entity_name}")

                # Step 3: Create evidence relations for all entities
                relations = []
                for entity_name in created_entity_names:
                    if entity_name != evidence_entity:
                        relations.append({
                            "source": entity_name,
                            "relationType": "evidence",
                            "target": evidence_entity
                        })

                if relations:
                    relations_payload = {"relations": relations}
                    result_rel = self.neo4j_connector.invoke_tool_by_name("create_relations", relations_payload)
                    logger.info(f"[agent] forced create_relations result: {TextUtils.truncate_text(result_rel, Config.LOG_TOOL_PREVIEW_MAX)}")

            logger.info(f"[agent] forced minimal write completed for doc={source_id} {chunk_id} - created {len(created_entity_names)} entities")
            return True, created_entity_names

        except Exception as exc:
            logger.error(f"[agent] forced write failed doc={source_id} {chunk_id}: {exc}", exc_info=True)
            return False, []

    # New helpers for relation retry dedup/normalization
    def _normalize_relation_for_retry(self, relation: Dict[str, Any]) -> Optional[Dict[str, str]]:
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
            return {"source": cast(str, source), "relationType": cast(str, relation_type), "target": cast(str, target)}
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
                self.neo4j_connector.invoke_tool_by_name("create_relations", {"relations": final_relations})
                logger.info(f"[agent] retried {len(final_relations)} domain relations at end of chunk")
            except Exception as exc:
                logger.warning(f"[agent] retrying relations failed: {exc}")

    def _generate_tool_calls_from_content(self, messages: List[Any], source_id: str, chunk_id: str, source_url: str, text: str) -> tuple[bool, List[str]]:
        """Generate comprehensive tool calls when the agent fails to use tools properly."""
        try:
            if not messages or not isinstance(messages[-1], AIMessage):
                return False, []

            last_ai_message = messages[-1]
            ai_content = getattr(last_ai_message, 'content', '')

            logger.info(f"[agent] generating comprehensive tool calls from AI content for doc={source_id} {chunk_id}")

            # Try to parse entities mentioned in the AI response
            entities_to_create = []
            created_names = []

            # Look for entities mentioned in AI response
            import re
            entity_patterns = [
                r"(?:person|individual|people?):?\s*([A-Z][a-zA-Z\s]{2,30})",
                r"(?:company|organization|corp|inc|ltd):?\s*([A-Z][a-zA-Z\s&]{2,40})",
                r"(?:location|city|country|place):?\s*([A-Z][a-zA-Z\s]{2,30})",
                r"(?:event|meeting|conference):?\s*([A-Z][a-zA-Z\s]{2,40})",
            ]

            entity_types = ["Person", "Organization", "Location", "Event"]

            for i, pattern in enumerate(entity_patterns):
                matches = re.findall(pattern, ai_content, re.IGNORECASE)
                for match in matches[:2]:  # Limit to 2 per type
                    clean_name = match.strip()
                    if len(clean_name) > 3 and clean_name not in created_names:
                        entities_to_create.append({
                            "name": clean_name,
                            "type": entity_types[i],
                            "observations": []
                        })
                        created_names.append(clean_name)

            # If no entities found in AI response, fall back to text analysis
            if not entities_to_create:
                # Simple extraction from original text
                words = text.split()
                for word in words[:30]:
                    clean_word = re.sub(r'[^\w\s]', '', word)
                    if (clean_word and clean_word[0].isupper() and 
                        len(clean_word) > 3 and clean_word not in created_names):
                        entities_to_create.append({
                            "name": clean_word,
                            "type": "Concept",
                            "observations": []
                        })
                        created_names.append(clean_word)
                        if len(entities_to_create) >= 3:
                            break

            # Always create Evidence entity
            evidence_name = f"Evidence {source_id}-{chunk_id}"
            entities_to_create.append({
                "name": evidence_name,
                "type": "Evidence",
                "observations": [f"srcId={source_id}", f"chunk={chunk_id}", f"url={source_url or ''}"]
            })
            created_names.append(evidence_name)

            if entities_to_create:
                # Create all entities
                entities_payload = {"entities": entities_to_create}
                result = self.neo4j_connector.invoke_tool_by_name("create_entities", entities_payload)
                logger.info(f"[agent] generated create_entities result: {TextUtils.truncate_text(result, Config.LOG_TOOL_PREVIEW_MAX)}")

                # Add observations to non-evidence entities
                for entity_name in created_names:
                    if entity_name != evidence_name:
                        obs_text = text if len(text) <= 600 else text[:600]
                        observations_payload = {"observations": [{"entityName": entity_name, "observations": [f"Context: {obs_text}"]}]}
                        self.neo4j_connector.invoke_tool_by_name("add_observations", observations_payload)

                # Create evidence relations
                relations = []
                for entity_name in created_names:
                    if entity_name != evidence_name:
                        relations.append({"source": entity_name, "relationType": "evidence", "target": evidence_name})

                if relations:
                    relations_payload = {"relations": relations}
                    self.neo4j_connector.invoke_tool_by_name("create_relations", relations_payload)

                logger.info(f"[agent] generated comprehensive tool calls completed for doc={source_id} {chunk_id} - created {len(created_names)} entities")
                return True, created_names

        except Exception as exc:
            logger.warning(f"[agent] failed to generate tool calls from content: {exc}")

        return False, []

    def process_chunk(self, source_id: str, chunk_id: str, source_url: str, text: str):
        """Process a single chunk of text."""
        logger.info(f"[chunk] start doc={source_id} {chunk_id} len={len(text)}")

        self._log_chunk_start_preview(source_id, chunk_id, text)

        wrote = False
        touched = []
        relations_to_retry = []

        try:
            # 1) Primary prompt run
            logger.info(f"[agent] starting primary prompt for doc={source_id} {chunk_id}")
            prompt = self._build_prompt(PROMPT_TMPL, source_id, chunk_id, source_url, text)
            messages = self._run_agent_sync(prompt)
            self._log_last_message_output(messages, source_id, chunk_id)

            wrote, touched, relations_to_retry = self._process_messages(messages)
            logger.info(f"[agent] primary prompt result doc={source_id} {chunk_id}: wrote={wrote}, touched_entities={len(touched)}")

            # 2) If no writes, try suggested calls from the last AI message
            if not wrote:
                logger.info(f"[agent] no writes detected, trying suggested tool calls for doc={source_id} {chunk_id}")
                exec_wrote, exec_touched, exec_rel = self._execute_suggested_calls_from_last_ai(messages)
                wrote = wrote or exec_wrote
                touched += exec_touched
                relations_to_retry += exec_rel
                logger.info(f"[agent] suggested calls result doc={source_id} {chunk_id}: wrote={exec_wrote}, additional_touched={len(exec_touched)}")

                # If still no writes, try generating tool calls from the AI response text
                if not exec_wrote and messages:
                    logger.info(f"[agent] no suggested calls worked, trying to generate tool calls from content for doc={source_id} {chunk_id}")
                    generated_wrote, generated_touched = self._generate_tool_calls_from_content(messages, source_id, chunk_id, source_url, text)
                    wrote = wrote or generated_wrote
                    touched += generated_touched
                    logger.info(f"[agent] generated calls result doc={source_id} {chunk_id}: wrote={generated_wrote}, additional_touched={len(generated_touched)}")

            # 3) Fallback prompt if still no writes
            # if not wrote and text and text.strip():
            #     logger.warning(f"[agent] no writes detected for doc={source_id} {chunk_id}; retrying with fallback prompt")
            #     fb_prompt = self._build_prompt(PROMPT_TMPL_FALLBACK, source_id, chunk_id, source_url, text)
            #     fb_messages = self._run_agent_sync(fb_prompt)
            #     self._log_last_message_output(fb_messages, source_id, chunk_id, prefix="fallback ")
            #
            #     fb_wrote, fb_touched, fb_rel = self._process_messages(fb_messages, prefix="fb-")
            #     wrote = wrote or fb_wrote
            #     touched += fb_touched
            #     relations_to_retry += fb_rel
            #     logger.info(f"[agent] fallback prompt result doc={source_id} {chunk_id}: wrote={fb_wrote}, additional_touched={len(fb_touched)}")
            #
            #     if not fb_wrote:
            #         logger.info(f"[agent] fallback prompt had no writes, trying suggested calls for doc={source_id} {chunk_id}")
            #         exec_wrote2, exec_touched2, exec_rel2 = self._execute_suggested_calls_from_last_ai(fb_messages)
            #         wrote = wrote or exec_wrote2
            #         touched += exec_touched2
            #         relations_to_retry += exec_rel2
            #         logger.info(f"[agent] fallback suggested calls result doc={source_id} {chunk_id}: wrote={exec_wrote2}, additional_touched={len(exec_touched2)}")

            # 4) Final programmatic fallback: ensure at least one write
            if not wrote and text and text.strip():
                logger.warning(f"[agent] forcing minimal write for doc={source_id} {chunk_id} (no AI writes succeeded)")
                fw_wrote, fw_touched = self._force_minimal_write(source_id, chunk_id, source_url, text)
                wrote = wrote or fw_wrote
                touched += fw_touched
                logger.info(f"[agent] forced write result doc={source_id} {chunk_id}: wrote={fw_wrote}, touched={len(fw_touched)}")

            # 5) Ensure evidence links for all touched entities
            final_touched = self._dedupe_preserve_order(touched)
            if final_touched:
                logger.info(f"[agent] creating evidence links for doc={source_id} {chunk_id}: {len(final_touched)} entities")
                self.ensure_evidence_links(final_touched, source_id, chunk_id, source_url)
            else:
                logger.warning(f"[agent] no entities to link to evidence for doc={source_id} {chunk_id}")

            # 6) Retry domain relations at the end
            if relations_to_retry:
                logger.info(f"[agent] retrying {len(relations_to_retry)} relations for doc={source_id} {chunk_id}")
                self._retry_relations(relations_to_retry)

            # Final success summary
            total_touched = len(final_touched)
            logger.info(f"[chunk] COMPLETED doc={source_id} {chunk_id}: wrote={wrote}, total_entities={total_touched}, relations_retried={len(relations_to_retry)}")

        except Exception as exc:
            logger.error(f"[chunk] FAILED doc={source_id} {chunk_id}: {exc}", exc_info=True)
            # Try to ensure at least basic tracking even on failure
            try:
                if not wrote and text and text.strip():
                    logger.info(f"[chunk] attempting emergency fallback for failed doc={source_id} {chunk_id}")
                    self._force_minimal_write(source_id, chunk_id, source_url, text)
            except Exception as fallback_exc:
                logger.error(f"[chunk] emergency fallback also failed for doc={source_id} {chunk_id}: {fallback_exc}")
            raise
