import contextlib
import os
import sys
import json
import time
import socket
import hashlib
import threading
import signal
import asyncio
from os import environ
from pathlib import Path
from typing import Optional, Dict, Any, List, cast
from urllib.parse import urlparse
import re

import httpx
from pydantic import BaseModel, Field, ConfigDict
import schedule
from loguru import logger

from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool

# Optional: import schema classes to construct exact tool payloads
try:
    from mcp_neo4j_memory.neo4j_memory import Entity as NeoEntity, ObservationAddition as NeoObservationAddition
except Exception:  # pragma: no cover - optional dependency
    NeoEntity = None  # type: ignore
    NeoObservationAddition = None  # type: ignore


# Environment Configuration
PAPERLESS_URL: str = os.getenv("PAPERLESS_URL", "http://paperless:8000")
PAPERLESS_TOKEN: Optional[str] = os.getenv("PAPERLESS_TOKEN")
PAPERLESS_TOKEN_FILE: Optional[str] = os.getenv("PAPERLESS_TOKEN_FILE")

OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://ollama:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama31-kb")

NEO4J_HOST: str = os.getenv("NEO4J_HOST", "host.docker.internal")
NEO4J_PORT: int = int(os.getenv("NEO4J_PORT", "7687"))
NEO4J_URL: Optional[str] = os.getenv("NEO4J_URL") or os.getenv("NEO4J_URI")
NEO4J_USER_ENV: Optional[str] = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER")
NEO4J_PASS_ENV: Optional[str] = os.getenv("NEO4J_PASSWORD") or os.getenv("NEO4J_PASS")
NEO4J_DATABASE: Optional[str] = os.getenv("NEO4J_DATABASE")

FORCE_REPROCESS: bool = str(os.getenv("FORCE_REPROCESS", "0")).lower() in {"1", "true", "yes"}

DEFAULT_MCP_CMD = "/app/.venv/bin/mcp-neo4j-memory"
MEMORY_MCP_CMD: str = os.getenv("MEMORY_MCP_CMD", DEFAULT_MCP_CMD)

STATE_PATH: Path = Path(os.getenv("STATE_PATH", "/data/state.json"))
VAULT_DIR: Path = Path(os.getenv("VAULT_DIR", "/data/obsidian"))
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "5000"))

# Logging controls
LOG_CHUNK_FULL: bool = str(os.getenv("LOG_CHUNK_FULL", "0")).lower() in {"1", "true", "yes"}
LOG_CHUNK_PREVIEW_MAX: int = int(os.getenv("LOG_CHUNK_PREVIEW_MAX", "2000"))
LOG_TOOL_PREVIEW_MAX: int = int(os.getenv("LOG_TOOL_PREVIEW_MAX", "1500"))
LOG_LLM_OUTPUT_MAX: int = int(os.getenv("LOG_LLM_OUTPUT_MAX", "4000"))

VAULT_DIR.mkdir(parents=True, exist_ok=True)

# Prompts
PROMPT_TMPL = """You: You are a professional journalist and language researcher who is able to distill key information from various sources and detect
relations between the observations and facts even if there are a time gaps between.
Task: Based on the text below, BUILD KNOWLEDGE into the Neo4j-based memory graph using the AVAILABLE TOOLS
(search_memories, find_memories_by_name, read_graph, create_entities, delete_entities, create_relations, delete_relations, add_observations, delete_observations).

Rules:
- Always perform: SEARCH → UPSERT → OBSERVATION. Use read_graph only if you need global context; do NOT stop after read_graph.
- Observations must be properties of the relevant entities (store observations on the entity itself, not on a generic document node).
- For each chunk, create exactly one Evidence entity and link all updated/created entities in this chunk to it via relationType="evidence".
  - Evidence entity spec: name="Evidence {SOURCE_ID}-{CHUNK_ID}", type="Evidence", observations=["srcId={SOURCE_ID}", "chunk={CHUNK_ID}", "url={SOURCE_URL}"].
- Use ONLY the fields supported by each tool:
  - create_entities expects: entities[{ name, type, observations }], where observations is a list of strings (can be empty []).
  - create_relations expects: relations[{ source, relationType, target }].
  - add_observations expects: observations[{ entityName, observations }], where observations is a list of strings.
- Do not duplicate: use search_memories/find_memories_by_name first; only create when not found.
- Canonize names: e.g. "OTP Bank Nyrt." → "OTP Bank".
- Date ISO: YYYY-MM-DD (if only year/month known, fill with 01). Money: value (number), unit (USD/HUF/EUR/$/Ft/€).
- Relation examples (not exhaustive): offers, announces, acquires, releases, located_in, headquartered_in, founded_on, ceo_of, part_of, depends_on,
  compatible_with, integrates_with, price_of, discounted_to, available_at, published_on, linked_to, cites, authored_by, version_of, supersedes, fixes,
  affected_by, related_to, deprecated_by, evidence.
- Continue calling tools until no more facts can be extracted, then write exactly: DONE.

Useful patterns (flat fields only; match tool schemas exactly):
- create_entities: {"entities":[{"name":"…","type":"Organization","observations":[]}]} 
- add_observations: {"observations":[{"entityName":"…","observations":["…"]}]}
- Evidence example:
  - create_entities: {"entities":[{"name":"Evidence {SOURCE_ID}-{CHUNK_ID}","type":"Evidence","observations":["srcId={SOURCE_ID}","chunk={CHUNK_ID}","url={SOURCE_URL}"]}]}
  - create_relations: {"relations":[{"source":"<EntityName>","relationType":"evidence","target":"Evidence {SOURCE_ID}-{CHUNK_ID}"}]}

TEXT:
<<<
{TEXT}
>>>

"""

PROMPT_TMPL_FALLBACK = """You are an extraction agent. From TEXT below, you MUST persist knowledge into the Neo4j memory using the tools.
Strictly follow: (1) search_memories/find_memories_by_name to avoid duplicates; (2) create_entities and/or create_relations for new facts; (3) add_observations on the correct entities; (4) create one Evidence entity for this chunk and link all updated/created entities to it with relationType="evidence".

Remember:
- Observations are stored on the entity nodes themselves.
- Evidence entity: name="Evidence {SOURCE_ID}-{CHUNK_ID}", type="Evidence", observations=["srcId={SOURCE_ID}", "chunk={CHUNK_ID}", "url={SOURCE_URL}"].
- create_entities: entities[{ name, type, observations }] where observations is a list of strings (can be empty).
- create_relations: relations[{ source, relationType, target }]
- add_observations: observations[{ entityName, observations }] where observations is a list of strings.
- Do NOT duplicate existing entities/relations; search first. Finish with DONE when nothing else to do.

TEXT:
<<<
{TEXT}
>>>
"""


class DocumentWork(BaseModel):
    model_config = ConfigDict(extra='forbid')

    doc_id: int = Field(..., description="Paperless document ID")
    source_url: str = Field("", description="Original source download URL, may be empty")
    chunks: List[str] = Field(default_factory=list, description="Chunks of extracted text")
    text_hash: str = Field(..., description="SHA256 hash of the full text for idempotency")
    doc: dict = Field(default_factory=dict, description="Original Paperless document payload")


class TextUtils:
    """Utility class for text processing operations."""

    @staticmethod
    def strip_code_fences(text: str) -> str:
        """Remove code fences from text."""
        if not text:
            return text
        return re.sub(r"```[a-zA-Z]*\s*\r?\n|```", "", text, flags=re.MULTILINE)

    @staticmethod
    def truncate_text(text: Any, limit: int) -> str:
        """Truncate text to specified limit with indicator."""
        try:
            s = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)
        except Exception:
            s = str(text)
        if limit <= 0 or len(s) <= limit:
            return s
        tail = len(s) - limit
        return f"{s[:limit]}... [truncated {tail} chars]"

    @staticmethod
    def chunk_text(text: str, max_chars: int = CHUNK_SIZE) -> List[str]:
        """Split text into chunks of a specified size."""
        return [text[i:i + max_chars] for i in range(0, len(text), max_chars)] if text else [""]


class JSONParser:
    """JSON parsing utilities for extracting structured data."""

    @staticmethod
    def is_valid_tool_call(obj: Any) -> bool:
        """Check if an object is a valid tool call with name and parameters."""
        return isinstance(obj, dict) and "name" in obj and "parameters" in obj

    @staticmethod
    def extract_json_object_at_position(text: str, start_pos: int) -> Optional[tuple[str, int]]:
        """Extract a complete JSON object starting at the given position."""
        depth = 0
        i = start_pos + 1  # Skip opening brace
        in_string = False
        escaped = False

        while i < len(text):
            char = text[i]

            if in_string:
                if escaped:
                    escaped = False
                elif char == '\\':
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '{':
                depth += 1
            elif char == '}':
                if depth == 0:
                    return text[start_pos:i+1], i
                depth -= 1
            elif char == '"':
                in_string = True
            i += 1

        return None

    @staticmethod
    def extract_json_array_at_position(text: str, start_pos: int) -> Optional[str]:
        """Extract a complete JSON array starting at the given position."""
        depth = 0
        i = start_pos + 1  # Skip opening bracket
        in_string = False
        escaped = False

        while i < len(text):
            char = text[i]

            if in_string:
                if escaped:
                    escaped = False
                elif char == '\\':
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '[':
                depth += 1
            elif char == ']':
                if depth == 0:
                    return text[start_pos:i+1]
                depth -= 1
            elif char == '"':
                in_string = True
            i += 1

        return None

    @classmethod
    def iter_json_objects(cls, text: str):
        """Yield JSON objects from string using balanced brace scanning."""
        i = 0
        n = len(text)

        while i < n:
            if text[i] == '{':
                json_obj = cls.extract_json_object_at_position(text, i)
                if json_obj:
                    yield json_obj[0]
                    i = json_obj[1] + 1
                else:
                    i += 1
            else:
                i += 1

    @classmethod
    def find_top_level_array(cls, text: str) -> Optional[str]:
        """Find and extract the first top-level JSON array in the string."""
        for i in range(len(text)):
            if text[i] == '[':
                array_content = cls.extract_json_array_at_position(text, i)
                if array_content:
                    return array_content
        return None


class ToolCallExtractor:
    """Extracts tool calls from LLM responses using multiple strategies."""

    @classmethod
    def try_direct_json_parse(cls, text: str) -> List[Dict[str, Any]]:
        """Try parsing text as direct JSON (object or array)."""
        with contextlib.suppress(Exception):
            parsed = json.loads(text)

            if JSONParser.is_valid_tool_call(parsed):
                return [parsed]

            if isinstance(parsed, list):
                valid_calls = [obj for obj in parsed if JSONParser.is_valid_tool_call(obj)]
                return valid_calls or []

        return []

    @classmethod
    def try_regex_extraction(cls, text: str) -> List[Dict[str, Any]]:
        """Extract tool calls using regex for simple cases (no nested braces)."""
        pattern = re.compile(
            r"\{\s*\"name\"\s*:\s*\"[^\"]+\"\s*,\s*\"parameters\"\s*:\s*\{[^{}]*\}\s*\}", 
            re.DOTALL
        )

        calls = []
        for match in pattern.finditer(text):
            block = match.group(0)
            with contextlib.suppress(Exception):
                obj = json.loads(block)
                if JSONParser.is_valid_tool_call(obj):
                    calls.append(obj)

        return calls

    @classmethod
    def try_balanced_brace_extraction(cls, text: str) -> List[Dict[str, Any]]:
        """Extract tool calls using balanced brace scanning for nested structures."""
        calls = []
        for block in JSONParser.iter_json_objects(text):
            with contextlib.suppress(Exception):
                obj = json.loads(block)
                if JSONParser.is_valid_tool_call(obj):
                    calls.append(obj)

        return calls

    @classmethod
    def try_array_extraction(cls, text: str) -> List[Dict[str, Any]]:
        """Try to find and parse a top-level JSON array."""
        with contextlib.suppress(Exception):
            arr_str = JSONParser.find_top_level_array(text)
            if arr_str:
                parsed_arr = json.loads(arr_str)
                if isinstance(parsed_arr, list):
                    valid_calls = [obj for obj in parsed_arr if JSONParser.is_valid_tool_call(obj)]
                    return valid_calls or []

        return []

    @classmethod
    def extract_tool_calls(cls, text: str) -> List[Dict[str, Any]]:
        """Extract tool calls from text using multiple parsing strategies."""
        if not text:
            return []

        cleaned = TextUtils.strip_code_fences(text)

        # Try each extraction strategy in order of preference
        strategies = [
            cls.try_direct_json_parse,
            cls.try_regex_extraction,
            cls.try_balanced_brace_extraction,
            cls.try_array_extraction
        ]

        for strategy in strategies:
            calls = strategy(cleaned)
            if calls:
                return calls[:10]

        return []


class ToolCallNormalizer:
    """Normalizes tool call parameters to ensure consistency."""

    @staticmethod
    def extract_name(value: Any) -> Any:
        """Extract name from dict or return value as-is."""
        return value.get("name") if isinstance(value, dict) and "name" in value else value

    @classmethod
    def extract_relation_entity_names(cls, relation: Dict[str, Any]) -> List[str]:
        """Extract entity names from a relation dict (source and target)."""
        names = []
        if isinstance(relation, dict):
            source = relation.get("source")
            target = relation.get("target")

            if isinstance(source, dict):
                source = source.get("name")
            if isinstance(target, dict):
                target = target.get("name")

            if isinstance(source, str):
                names.append(source)
            if isinstance(target, str):
                names.append(target)
        return names

    @classmethod
    def normalize_single_relation(cls, relation: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a single relation dictionary."""
        normalized = dict(relation)

        # Handle field name variations
        if "subject" in normalized and "source" not in normalized:
            normalized["source"] = cls.extract_name(normalized.pop("subject"))
        if "object" in normalized and "target" not in normalized:
            normalized["target"] = cls.extract_name(normalized.pop("object"))
        if "predicate" in normalized and "relationType" not in normalized:
            normalized["relationType"] = normalized.pop("predicate")

        normalized["source"] = cls.extract_name(normalized.get("source"))
        normalized["target"] = cls.extract_name(normalized.get("target"))
        return normalized

    @classmethod
    def normalize_relations_params(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize parameters for create_relations tool."""
        relations = params.get("relations")

        if not relations:
            # Try to build from individual fields
            maybe = {k: params.get(k) for k in ("source", "predicate", "relationType", "target", "when", "evidence", "confidence", "sourceId", "chunkId", "sourceUrl") if k in params}
            if any(maybe.values()):
                relations = [maybe]

        normalized_relations = [
            cls.normalize_single_relation(r)
            for r in relations or []
            if isinstance(r, dict)
        ]

        result = {k: v for k, v in params.items() if k != "relations"}
        result["relations"] = normalized_relations
        return result

    @classmethod
    def normalize_observation(cls, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a single observation dictionary."""
        normalized = dict(observation)

        # Normalize entity name field
        entity_name_mappings = [
            ("entity", "entityName"),
            ("entity_name", "entityName"),
            ("name", "entityName")
        ]

        for old_key, new_key in entity_name_mappings:
            if old_key in normalized and new_key not in normalized:
                normalized[new_key] = cls.extract_name(normalized.pop(old_key))

        # Normalize observations field
        if "observations" not in normalized or not isinstance(normalized.get("observations"), list):
            if "text" in normalized and normalized["text"]:
                normalized["observations"] = [str(normalized.pop("text"))]
            else:
                normalized["observations"] = []

        return normalized

    @classmethod
    def extract_observations_list(cls, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract observations list from parameters."""
        if observations := params.get("observations"):
            return observations

        if "observation" in params and isinstance(params["observation"], dict):
            return [params["observation"]]

        if any(k in params for k in ("entityName", "text", "observations")):
            observations_list = cls.build_observations_list(params)
            return [{"entityName": params.get("entityName"), "observations": observations_list}]

        return []

    @classmethod
    def build_observations_list(cls, params: Dict[str, Any]) -> List[str]:
        """Build observations list from parameters."""
        if isinstance(params.get("observations"), list):
            return [str(x) for x in params["observations"]]
        elif "text" in params and params["text"]:
            return [str(params["text"])]
        return []

    @classmethod
    def normalize_observations_params(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize parameters for add_observations tool."""
        observations = cls.extract_observations_list(params)
        normalized_observations = [cls.normalize_observation(o) for o in observations or [] if isinstance(o, dict)]

        result = {k: v for k, v in params.items() if k not in {"observations", "observation", "text"}}
        result["observations"] = normalized_observations
        return result

    @classmethod
    def normalize_single_entity(cls, entity: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalize a single entity dictionary."""
        normalized = {k: v for k, v in entity.items() if k in {"name", "type", "observations"}}

        if "name" not in normalized:
            return None

        if "observations" not in normalized or not isinstance(normalized.get("observations"), list):
            normalized["observations"] = []

        return normalized

    @classmethod
    def extract_entities_from_params(cls, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract entities list from various parameter formats."""
        if entities := params.get("entities"):
            return entities

        if "entity" in params and isinstance(params["entity"], dict):
            return [params["entity"]]

        if any(k in params for k in ("name", "type")):
            entity = {k: params.get(k) for k in ("name", "type") if k in params}
            return [entity]

        if cls.has_valid_observations(params):
            first_observation = params["observations"][0]
            entity = {
                "name": first_observation["entityName"], 
                "type": params.get("type", "Thing")
            }
            return [entity]

        return []

    @classmethod
    def has_valid_observations(cls, params: Dict[str, Any]) -> bool:
        """Check if parameters contain valid observations with entity name."""
        observations = params.get("observations")
        if not isinstance(observations, list) or not observations:
            return False

        first_observation = observations[0]
        return isinstance(first_observation, dict) and "entityName" in first_observation

    @classmethod
    def normalize_entities_params(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize parameters for the create_entities tool."""
        entities = cls.extract_entities_from_params(params)
        normalized_entities = [
            normalized_entity 
            for entity in entities or []
            if isinstance(entity, dict) and (normalized_entity := cls.normalize_single_entity(entity))
        ]

        result = {k: v for k, v in params.items() if k != "entities"}
        result["entities"] = normalized_entities
        return result

    @classmethod
    def normalize_params(cls, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize parameters for different tools."""
        normalizers = {
            "create_relations": cls.normalize_relations_params,
            "add_observations": cls.normalize_observations_params,
            "create_entities": cls.normalize_entities_params,
        }

        normalizer = normalizers.get(tool_name)
        return normalizer(params) if normalizer else params


class Neo4jMemoryConnector:
    """Manages connection to Neo4j memory via MCP adapters."""

    def __init__(self):
        self._tools_cache: List[BaseTool] = []
        self._tools_loaded: bool = False

    @staticmethod
    def neo4j_host_port_from_url(url: str) -> Optional[tuple[str, int]]:
        """Extract host and port from Neo4j URL."""
        with contextlib.suppress(Exception):
            p = urlparse(url)
            if p.hostname:
                return p.hostname, p.port or 7687
        return None

    @staticmethod
    def build_stdio_server_config() -> Dict[str, Any]:
        """Build MCP stdio server configuration."""
        base_cmd = MEMORY_MCP_CMD or DEFAULT_MCP_CMD
        db_url = NEO4J_URL or f"bolt://{NEO4J_HOST}:{NEO4J_PORT}"
        args: List[str] = ["--db-url", db_url]

        if NEO4J_USER_ENV:
            args += ["--username", NEO4J_USER_ENV]
        if NEO4J_PASS_ENV:
            args += ["--password", NEO4J_PASS_ENV]
        if NEO4J_DATABASE:
            args += ["--database", NEO4J_DATABASE]

        logger.info(f"[mcp] stdio cmd: {base_cmd} args={args}")
        return {
            "neo4j": {
                "command": base_cmd,
                "args": args,
                "transport": "stdio",
            }
        }

    @classmethod
    async def load_mcp_tools(cls) -> List[BaseTool]:
        """Load MCP tools from configuration."""
        config = cls.build_stdio_server_config()
        client = MultiServerMCPClient(config)
        tools = await client.get_tools()
        logger.info(f"[mcp] loaded tools via adapter: {[t.name for t in tools]}")

        # Debug: log tool schemas
        for tool in tools:
            try:
                schema = None
                args_schema = getattr(tool, "args_schema", None)
                if args_schema is not None:
                    if hasattr(args_schema, "model_json_schema"):
                        schema = args_schema.model_json_schema()
                    elif hasattr(args_schema, "schema"):
                        schema = args_schema.schema()
                if isinstance(schema, dict):
                    required = schema.get("required") or []
                    logger.info(f"[mcp] tool schema name={tool.name} required={required}")
            except Exception:
                logger.debug(f"[mcp] tool schema dump failed for {tool.name}")

        return tools

    async def ensure_mcp_tools(self) -> List[BaseTool]:
        """Ensure MCP tools are loaded and cached."""
        if not self._tools_loaded:
            self._tools_cache = await self.load_mcp_tools()
            self._tools_loaded = True
        return self._tools_cache

    def get_tools_by_name_sync(self) -> Dict[str, BaseTool]:
        """Get tools dictionary by name (synchronous)."""
        try:
            tools = asyncio.run(self.ensure_mcp_tools())
        except RuntimeError:
            tools = self._tools_cache
        return {t.name: t for t in tools}

    @staticmethod
    def try_async_invoke(tool: BaseTool, params: Dict[str, Any]) -> Any:
        """Try to invoke the tool using async ainvoke method."""
        if not hasattr(tool, 'ainvoke'):
            return None

        try:
            return asyncio.run(tool.ainvoke(params))
        except Exception:
            return None

    @staticmethod
    def try_sync_invoke(tool: BaseTool, params: Dict[str, Any]) -> Any:
        """Try to invoke the tool using sync invoke method."""
        if not hasattr(tool, 'invoke'):
            return None

        try:
            return tool.invoke(params)
        except Exception as exc:
            if "async" in str(exc).lower() or "await" in str(exc).lower():
                return Neo4jMemoryConnector.try_async_invoke(tool, params)
            return None

    @staticmethod
    def try_legacy_run(tool: BaseTool, params: Dict[str, Any]) -> Any:
        """Try to invoke the tool using legacy run method."""
        if not hasattr(tool, 'run'):
            return None

        try:
            return tool.run(**params)
        except Exception:
            return None

    @staticmethod
    def try_callable_fallback(tool: BaseTool, params: Dict[str, Any]) -> Any:
        """Try to invoke the tool as a callable."""
        if not callable(tool):
            raise RuntimeError(f"Tool '{tool}' cannot be invoked with any known method")

        try:
            return tool(**params)
        except Exception as exc:
            raise RuntimeError(f"Failed to invoke tool '{tool}': {exc}") from exc

    @classmethod
    def try_tool_invocation_methods(cls, tool: BaseTool, params: Dict[str, Any]) -> Any:
        """Try different invocation methods for a tool in order of preference."""
        result = cls.try_async_invoke(tool, params)
        if result is not None:
            return result

        result = cls.try_sync_invoke(tool, params)
        if result is not None:
            return result

        result = cls.try_legacy_run(tool, params)
        return result if result is not None else cls.try_callable_fallback(tool, params)

    def invoke_tool_by_name(self, name: str, params: Dict[str, Any]) -> Any:
        """Invoke a tool by name with the given parameters."""
        tools = self.get_tools_by_name_sync()
        if name not in tools:
            raise ValueError(f"Tool '{name}' not found")

        tool = tools[name]
        normalized_params = ToolCallNormalizer.normalize_params(name, params)
        return self.try_tool_invocation_methods(tool, normalized_params)


class PaperlessProcessor:
    """Handles document retrieval and processing from Paperless."""

    @staticmethod
    def wait_for_token(timeout_seconds: int = 0) -> str:
        """Wait for Paperless authentication token."""
        global PAPERLESS_TOKEN
        if PAPERLESS_TOKEN:
            return PAPERLESS_TOKEN

        token_path = PAPERLESS_TOKEN_FILE
        if not token_path:
            raise RuntimeError("Neither PAPERLESS_TOKEN nor PAPERLESS_TOKEN_FILE is specified.")

        logger.info(f"[bootstrap] Watching token: {token_path}")
        deadline = (time.time() + timeout_seconds) if timeout_seconds > 0 else None

        while True:
            with contextlib.suppress(Exception):
                if os.path.isfile(token_path) and os.path.getsize(token_path) > 0:
                    content = Path(token_path).read_text(encoding="utf-8").strip()
                    if content and content.upper() != "PENDING":
                        PAPERLESS_TOKEN = content
                        logger.info("[bootstrap] Paperless token read.")
                        return content

            if deadline and time.time() > deadline:
                raise RuntimeError("Token not available within the specified time.")
            time.sleep(2)

    @classmethod
    def get_headers(cls) -> dict:
        """Get authorization headers for Paperless API."""
        token = cls.wait_for_token()
        return {"Authorization": f"Token {token}"}

    @classmethod
    def iter_documents(cls):
        """Iterate through all documents from Paperless API."""
        url = f"{PAPERLESS_URL}/api/documents/?ordering=id"
        headers = cls.get_headers()

        while url:
            with httpx.Client(timeout=30) as client:
                response = client.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()
                yield from data.get("results", [])
                url = data.get("next")

    @classmethod
    def get_document(cls, doc_id: int) -> dict:
        """Get detailed document data by ID."""
        url = f"{PAPERLESS_URL}/api/documents/{doc_id}/"
        headers = cls.get_headers()

        with httpx.Client(timeout=30) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()
            return response.json()

    @staticmethod
    def extract_text(document: dict) -> str:
        """Extract text content from document."""
        return (document.get("content") or "").strip()

    @staticmethod
    def write_obsidian_note(document: dict, chunk_index: int, text: str):
        """Write chunk to Obsidian vault (optional)."""
        try:
            slug = f"{document['id']}_c{chunk_index+1}"
            meta = {
                "title": document.get("title") or slug,
                "created": document.get("created"),
                "source": document.get("download_url"),
                "paperless_id": document["id"],
                "chunk": chunk_index + 1
            }
            body = "---\n" + json.dumps(meta, ensure_ascii=False, indent=2) + "\n---\n\n" + text + "\n"
            (VAULT_DIR / f"{slug}.md").write_text(body, encoding="utf-8")
            logger.info(f"[obsidian] wrote: {slug}.md ({len(text)} chars)")
        except Exception:
            logger.exception("Obsidian write error")


class StateManager:
    """Manages processing state and idempotency checks."""

    def __init__(self, state_path: Path = STATE_PATH):
        self.state_path = state_path
        self.state: Dict[str, Any] = {"last_id": 0, "hashes": {}}
        self._load_state()

    def _load_state(self):
        """Load state from file if it exists."""
        if self.state_path.exists():
            try:
                self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            except Exception:
                pass

    def save_state(self):
        """Save current state to file."""
        with contextlib.suppress(Exception):
            self.state_path.write_text(json.dumps(self.state), encoding="utf-8")

    def should_process_document(self, doc_id: int) -> bool:
        """Check if document should be processed based on state."""
        if FORCE_REPROCESS:
            return True
        return doc_id > self.state.get("last_id", 0)

    def is_document_changed(self, doc_id: int, text_hash: str) -> bool:
        """Check if document content has changed."""
        if FORCE_REPROCESS:
            return True
        return self.state["hashes"].get(str(doc_id)) != text_hash

    def update_document_state(self, doc_id: int, text_hash: str):
        """Update state for processed document."""
        self.state["hashes"][str(doc_id)] = text_hash
        self.state["last_id"] = doc_id
        self.save_state()
        logger.info(f"[state] saved id={doc_id} hash_prefix={text_hash[:8]}")

    def advance_last_id(self, doc_id: int):
        """Advance last processed ID (for skipped documents)."""
        self.state["last_id"] = doc_id
        self.save_state()
        logger.info(f"[state] advanced last_id={doc_id}")


class ServiceBootstrapper:
    """Handles service availability checks and bootstrap process."""

    @staticmethod
    def wait_for_neo4j(host: str, port: int, timeout: int = 240):
        """Wait for Neo4j to become available."""
        logger.info(f"[bootstrap] Waiting for Neo4j at {host}:{port} ...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                with socket.create_connection((host, port), timeout=2):
                    logger.info("[bootstrap] Neo4j available.")
                    return
            except OSError:
                time.sleep(2)

        raise RuntimeError("Neo4j not available.")

    @staticmethod
    def wait_for_http_service(url: str, timeout: int = 240):
        """Wait for HTTP service to become available."""
        logger.info(f"[bootstrap] Waiting for HTTP service: {url}")
        start_time = time.time()

        while time.time() - start_time < timeout:
            with contextlib.suppress(Exception):
                with httpx.Client(timeout=5) as client:
                    response = client.get(url)
                    if 200 <= response.status_code < 500:
                        logger.info(f"[bootstrap] Available: {url}")
                        return
            time.sleep(2)

        raise RuntimeError(f"Service not available: {url}")

    @classmethod
    def resolve_neo4j_host_port(cls) -> tuple[str, int]:
        """Resolve Neo4j host and port from configuration."""
        if NEO4J_URL:
            host_port = Neo4jMemoryConnector.neo4j_host_port_from_url(NEO4J_URL)
            if host_port:
                return host_port
        return NEO4J_HOST, NEO4J_PORT

    @classmethod
    def bootstrap_all_services(cls):
        """Bootstrap all required services."""
        cls.wait_for_http_service(PAPERLESS_URL, timeout=600)
        host, port = cls.resolve_neo4j_host_port()
        cls.wait_for_neo4j(host, port, timeout=600)
        cls.wait_for_http_service(f"{OLLAMA_URL}/api/tags", timeout=600)
        PaperlessProcessor.get_headers()  # Validate token
        logger.info("[bootstrap] Services ready.")


class AgentOrchestrator:
    """Orchestrates AI agent processing with Neo4j knowledge graph."""

    def __init__(self, neo4j_connector: Neo4jMemoryConnector):
        self.neo4j_connector = neo4j_connector

    async def run_agent(self, prompt: str) -> Dict[str, Any]:
        """Run AI agent with given prompt."""
        tools = await self.neo4j_connector.ensure_mcp_tools()
        model = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_URL)
        graph = create_react_agent(model, tools)
        state: Any = {"messages": prompt}
        return await graph.ainvoke(cast(Any, state))

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

    def extract_entities_from_calls(self, calls: List[Dict[str, Any]]) -> List[str]:
        """Extract entity names from tool calls."""
        names = []
        for call in calls or []:
            try:
                name = call.get("name")
                params = call.get("parameters") or {}

                if name == "add_observations":
                    observations = params.get("observations") or []
                    for observation in observations:
                        entity_name = observation.get("entityName") if isinstance(observation, dict) else None
                        if isinstance(entity_name, dict):
                            entity_name = entity_name.get("name")
                        if isinstance(entity_name, str):
                            names.append(entity_name)

                elif name == "create_entities":
                    entities = params.get("entities") or []
                    for entity in entities:
                        entity_name = entity.get("name") if isinstance(entity, dict) else None
                        if isinstance(entity_name, str):
                            names.append(entity_name)

                elif name == "create_relations":
                    relations = params.get("relations") or []
                    for relation in relations:
                        names.extend(ToolCallNormalizer.extract_relation_entity_names(relation))
            except Exception:
                continue

        # Deduplicate while preserving order
        seen = set()
        return [name for name in names if name not in seen and not seen.add(name)]

    def execute_tool_calls(self, calls: List[Dict[str, Any]]) -> tuple[bool, List[str]]:
        """Execute tool calls in proper order."""
        if not calls:
            return False, []

        wrote = False
        touched = []

        # Partition calls to ensure proper execution order
        create_entities_calls = [c for c in calls if c.get("name") == "create_entities"]
        add_observations_calls = [c for c in calls if c.get("name") == "add_observations"]
        create_relations_calls = [c for c in calls if c.get("name") == "create_relations"]
        other_calls = [c for c in calls if c.get("name") not in {"create_entities", "add_observations", "create_relations"}]

        ordered_calls = create_entities_calls + add_observations_calls + other_calls + create_relations_calls

        for i, call in enumerate(ordered_calls, start=1):
            name = call.get("name")
            params = call.get("parameters")

            if not isinstance(name, str) or not isinstance(params, dict):
                continue

            try:
                logger.info(f"[agent] executing suggested tool #{i}: {name} args={TextUtils.truncate_text(params, LOG_TOOL_PREVIEW_MAX)}")
                result = self.neo4j_connector.invoke_tool_by_name(name, params)
                logger.info(f"[agent] suggested tool result #{i} {name}: {TextUtils.truncate_text(result, LOG_TOOL_PREVIEW_MAX)}")

                if name in {"create_entities", "create_relations", "add_observations"} and not (isinstance(result, str) and str(result).lower().startswith("error")):
                    wrote = True

                # Collect touched entities
                if name == "create_entities":
                    entities = (params or {}).get("entities") or []
                    for entity in entities:
                        entity_name = entity.get("name") if isinstance(entity, dict) else None
                        if isinstance(entity_name, str):
                            touched.append(entity_name)

                elif name == "add_observations":
                    observations = (params or {}).get("observations") or []
                    for observation in observations:
                        entity_name = observation.get("entityName") if isinstance(observation, dict) else None
                        if isinstance(entity_name, dict):
                            entity_name = entity_name.get("name")
                        if isinstance(entity_name, str):
                            touched.append(entity_name)

                elif name == "create_relations":
                    relations = (params or {}).get("relations") or []
                    for relation in relations:
                        touched.extend(ToolCallNormalizer.extract_relation_entity_names(relation))

            except Exception as exc:
                logger.warning(f"[agent] suggested tool error {name}: {exc}")

        # Deduplicate touched entities
        seen = set()
        touched_unique = [name for name in touched if isinstance(name, str) and name and (name not in seen and not seen.add(name))]
        return wrote, touched_unique

    def ensure_evidence_links(self, entity_names: List[str], source_id: str, chunk_id: str, source_url: str):
        """Ensure evidence entity exists and is linked to all touched entities."""
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
        relations = [{"source": name, "relationType": "evidence", "target": evidence_name} for name in sources]

        if relations:
            try:
                self.neo4j_connector.invoke_tool_by_name("create_relations", {"relations": relations})
                logger.info(f"[agent] linked {len(relations)} entities to evidence {evidence_name}")
            except Exception as exc:
                logger.warning(f"[agent] evidence relation creation failed: {exc}")

    def process_chunk(self, source_id: str, chunk_id: str, source_url: str, text: str):
        """Process a single chunk of text."""
        logger.info(f"[chunk] start doc={source_id} {chunk_id} len={len(text)}")

        if LOG_CHUNK_FULL:
            logger.info(f"[chunk] text doc={source_id} {chunk_id}:\n{text}")
        else:
            logger.info(f"[chunk] preview doc={source_id} {chunk_id}:\n{TextUtils.truncate_text(text, LOG_CHUNK_PREVIEW_MAX)}")

        try:
            # Build prompt
            prompt = (PROMPT_TMPL
                      .replace("{SOURCE_ID}", source_id)
                      .replace("{CHUNK_ID}", chunk_id)
                      .replace("{SOURCE_URL}", source_url or "")
                      .replace("{TEXT}", text))

            # Run agent
            result = asyncio.run(self.run_agent(prompt))
            messages = result.get("messages") if isinstance(result, dict) else []

            if messages:
                last_message = messages[-1]
                logger.info(f"[agent] output doc={source_id} {chunk_id}:\n{TextUtils.truncate_text(getattr(last_message, 'content', ''), LOG_LLM_OUTPUT_MAX)}")

            wrote = False
            touched = []
            relations_to_retry = []

            # Process messages to detect writes and collect touched entities
            for i, message in enumerate(messages or [], start=1):
                try:
                    if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
                        for tool_call in message.tool_calls:
                            logger.info(f"[agent] ai#{i} tool_call name={tool_call.get('name')} args={TextUtils.truncate_text(tool_call.get('args'), LOG_TOOL_PREVIEW_MAX)}")

                    elif isinstance(message, ToolMessage):
                        logger.info(f"[agent] tool#{i} name={message.name} result={TextUtils.truncate_text(message.content, LOG_TOOL_PREVIEW_MAX)}")

                        if message.name in {"create_entities", "create_relations", "add_observations", "delete_entities", "delete_relations", "delete_observations"}:
                            content = (message.content or "") if isinstance(message.content, str) else json.dumps(message.content)
                            if not str(content).lower().startswith("error"):
                                wrote = True

                        # Collect touched entities
                        with contextlib.suppress(Exception):
                            raw = message.content
                            data = None

                            if isinstance(raw, str):
                                with contextlib.suppress(Exception):
                                    data = json.loads(raw)
                            elif isinstance(raw, (dict, list)):
                                data = raw

                            if message.name == "add_observations" and isinstance(data, list):
                                touched += [d.get("entityName") for d in data if isinstance(d, dict) and isinstance(d.get("entityName"), str)]

                            if message.name == "create_entities" and isinstance(data, list):
                                touched += [d.get("name") for d in data if isinstance(d, dict) and isinstance(d.get("name"), str)]

                            if message.name == "create_relations" and isinstance(data, list):
                                for relation in data:
                                    if isinstance(relation, dict):
                                        relations_to_retry.append(relation)

                except Exception:
                    logger.info(f"[agent] msg#{i} (unparsable)")

            # If no writes detected, try to extract and execute suggested tool calls
            if not wrote and messages and isinstance(messages[-1], AIMessage):
                last_ai = cast(AIMessage, messages[-1])
                suggested_calls = ToolCallExtractor.extract_tool_calls(getattr(last_ai, "content", ""))

                if suggested_calls:
                    exec_wrote, exec_touched = self.execute_tool_calls(suggested_calls)
                    wrote = exec_wrote or wrote
                    touched += exec_touched
                    relations_to_retry += self.extract_relations_from_calls(suggested_calls)

            # Fallback prompt if still no writes
            if not wrote and text and text.strip():
                logger.warning(f"[agent] no writes detected for doc={source_id} {chunk_id}; retrying with fallback prompt")

                fallback_prompt = (PROMPT_TMPL_FALLBACK
                                   .replace("{SOURCE_ID}", source_id)
                                   .replace("{CHUNK_ID}", chunk_id)
                                   .replace("{SOURCE_URL}", source_url or "")
                                   .replace("{TEXT}", text))

                result2 = asyncio.run(self.run_agent(fallback_prompt))
                messages2 = result2.get("messages") if isinstance(result2, dict) else []

                if messages2:
                    last_message2 = messages2[-1]
                    logger.info(f"[agent] fallback output doc={source_id} {chunk_id}:\n{TextUtils.truncate_text(getattr(last_message2, 'content', ''), LOG_LLM_OUTPUT_MAX)}")

                # Process fallback messages
                for i, message in enumerate(messages2 or [], start=1):
                    try:
                        if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
                            for tool_call in message.tool_calls:
                                logger.info(f"[agent] fb-ai#{i} tool_call name={tool_call.get('name')} args={TextUtils.truncate_text(tool_call.get('args'), LOG_TOOL_PREVIEW_MAX)}")

                        elif isinstance(message, ToolMessage):
                            logger.info(f"[agent] fb-tool#{i} name={message.name} result={TextUtils.truncate_text(message.content, LOG_TOOL_PREVIEW_MAX)}")

                            if message.name in {"create_entities", "create_relations", "add_observations"}:
                                content = (message.content or "") if isinstance(message.content, str) else json.dumps(message.content)
                                if not str(content).lower().startswith("error"):
                                    wrote = True

                            # Collect touched entities from fallback
                            with contextlib.suppress(Exception):
                                raw = message.content
                                data = None

                                if isinstance(raw, str):
                                    with contextlib.suppress(Exception):
                                        data = json.loads(raw)
                                elif isinstance(raw, (dict, list)):
                                    data = raw

                                if message.name == "add_observations" and isinstance(data, list):
                                    touched += [d.get("entityName") for d in data if isinstance(d, dict) and isinstance(d.get("entityName"), str)]

                                if message.name == "create_entities" and isinstance(data, list):
                                    touched += [d.get("name") for d in data if isinstance(d, dict) and isinstance(d.get("name"), str)]

                                if message.name == "create_relations" and isinstance(data, list):
                                    for relation in data:
                                        if isinstance(relation, dict):
                                            relations_to_retry.append(relation)

                    except Exception:
                        logger.info(f"[agent] fb-msg#{i} (unparsable)")

                # Try suggested calls from fallback
                if not wrote and messages2 and isinstance(messages2[-1], AIMessage):
                    suggested_calls2 = ToolCallExtractor.extract_tool_calls(getattr(messages2[-1], "content", ""))
                    if suggested_calls2:
                        exec_wrote2, exec_touched2 = self.execute_tool_calls(suggested_calls2)
                        wrote = exec_wrote2 or wrote
                        touched += exec_touched2
                        relations_to_retry += self.extract_relations_from_calls(suggested_calls2)

            # Final programmatic fallback: ensure at least one write
            if not wrote and text and text.strip():
                try:
                    doc_entity = f"Document {source_id}"
                    logger.warning(f"[agent] forcing minimal write for doc={source_id} {chunk_id}")

                    # Create Document + Evidence entities
                    try:
                        entities_payload = {"entities": [
                            {"name": doc_entity, "type": "Document", "observations": []},
                            {"name": f"Evidence {source_id}-{chunk_id}", "type": "Evidence", 
                             "observations": [f"srcId={source_id}", f"chunk={chunk_id}", f"url={source_url or ''}"]}
                        ]}
                        result_create = self.neo4j_connector.invoke_tool_by_name("create_entities", entities_payload)
                        logger.info(f"[agent] forced create_entities result: {TextUtils.truncate_text(result_create, LOG_TOOL_PREVIEW_MAX)}")
                    except Exception as exc:
                        logger.warning(f"[agent] forced create_entities failed: {exc}")

                    # Add observations to Document
                    try:
                        obs_text = text if len(text) <= 1500 else text[:1500]
                        observations_payload = {"observations": [{"entityName": doc_entity, "observations": [obs_text]}]}
                        result_obs = self.neo4j_connector.invoke_tool_by_name("add_observations", observations_payload)
                        logger.info(f"[agent] forced add_observations result: {TextUtils.truncate_text(result_obs, LOG_TOOL_PREVIEW_MAX)}")
                    except Exception as exc:
                        logger.warning(f"[agent] forced add_observations failed: {exc}")

                    # Create evidence relation
                    try:
                        self.neo4j_connector.invoke_tool_by_name("create_relations", {"relations": [{
                            "source": doc_entity,
                            "relationType": "evidence",
                            "target": f"Evidence {source_id}-{chunk_id}"
                        }]})
                    except Exception as exc:
                        logger.warning(f"[agent] forced evidence relation failed: {exc}")

                    touched.append(doc_entity)
                    wrote = True

                except Exception as exc:
                    logger.error(f"[agent] forced write failed doc={source_id} {chunk_id}: {exc}")

            # Ensure evidence links for all touched entities
            if touched:
                seen = set()
                touched_unique = [name for name in touched if isinstance(name, str) and name and (name not in seen and not seen.add(name))]
                self.ensure_evidence_links(touched_unique, source_id, chunk_id, source_url)

            # Retry domain relations at the end
            if relations_to_retry:
                dedup_key = set()
                final_relations = []

                for relation in relations_to_retry:
                    if not isinstance(relation, dict):
                        continue

                    source = relation.get("source")
                    target = relation.get("target")
                    relation_type = relation.get("relationType") or relation.get("predicate")

                    # Handle dict endpoints
                    if isinstance(source, dict):
                        source = source.get("name")
                    if isinstance(target, dict):
                        target = target.get("name")

                    key = (source, relation_type, target)
                    if all(isinstance(x, str) and x for x in key) and key not in dedup_key:
                        final_relations.append({"source": source, "relationType": relation_type, "target": target})
                        dedup_key.add(key)

                if final_relations:
                    try:
                        self.neo4j_connector.invoke_tool_by_name("create_relations", {"relations": final_relations})
                        logger.info(f"[agent] retried {len(final_relations)} domain relations at end of chunk")
                    except Exception as exc:
                        logger.warning(f"[agent] retrying relations failed: {exc}")

            logger.info(f"[agent] done for doc={source_id} {chunk_id}; messages={len(messages) if messages else 0}")

        except Exception as exc:
            logger.error(f"[agent] failure doc={source_id} {chunk_id}: {exc}")
            raise


class DocumentProcessor:
    """Main ETL pipeline for processing documents."""

    def __init__(self):
        self.neo4j_connector = Neo4jMemoryConnector()
        self.agent_orchestrator = AgentOrchestrator(self.neo4j_connector)
        self.paperless_processor = PaperlessProcessor()
        self.state_manager = StateManager()
        self.bootstrapper = ServiceBootstrapper()

    def prepare_document_work(self, document: dict) -> Optional[DocumentWork]:
        """Prepare document work unit with validation and chunking."""
        try:
            doc_id = int(document.get("id", 0))
        except Exception:
            return None

        logger.info(f"[doc] consider id={doc_id} force={FORCE_REPROCESS}")

        if not self.state_manager.should_process_document(doc_id):
            logger.info(f"[doc] skip id={doc_id} last_id={self.state_manager.state.get('last_id', 0)}")
            return None

        try:
            detailed = self.paperless_processor.get_document(doc_id)
        except Exception as exc:
            logger.error(f"[doc] detail fetch failed id={doc_id}: {exc}")
            return None

        text = self.paperless_processor.extract_text(detailed)

        if not text:
            # Try fallback from metadata
            fallback = " ".join([
                str(detailed.get("title") or ""),
                str(detailed.get("notes") or ""),
                str(detailed.get("original_filename") or ""),
                str(detailed.get("created") or ""),
            ]).strip()

            if fallback:
                logger.info(f"[doc] empty OCR content; using metadata fallback id={doc_id}")
            text = fallback

        if not text:
            logger.info(f"[doc] no usable text id={doc_id}; advancing state")
            self.state_manager.advance_last_id(doc_id)
            return None

        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        if not self.state_manager.is_document_changed(doc_id, text_hash):
            logger.info(f"[doc] unchanged hash; skip id={doc_id}")
            self.state_manager.advance_last_id(doc_id)
            return None

        chunks = TextUtils.chunk_text(text)
        logger.info(f"[doc] prepared id={doc_id} chunks={len(chunks)} first_len={len(chunks[0]) if chunks else 0}")

        source_url = str(detailed.get("download_url") or "")
        return DocumentWork(doc_id=doc_id, source_url=source_url, chunks=chunks, text_hash=text_hash, doc=detailed)

    def run_downstream_steps(self, work: DocumentWork):
        """Execute downstream processing steps for document work."""
        for chunk_index, chunk_text in enumerate(work.chunks):
            source_id = str(work.doc_id)
            chunk_id = f"c{chunk_index+1}"

            try:
                self.agent_orchestrator.process_chunk(source_id, chunk_id, work.source_url, chunk_text)
            except Exception as exc:
                logger.warning(f"Agent/MCP error doc {work.doc_id} {chunk_id}: {exc}")

            # Optional Obsidian export
            if str(environ.get("OBSIDIAN_EXPORT", False)).lower() in {"1", "true", "yes"}:
                try:
                    self.paperless_processor.write_obsidian_note(work.doc, chunk_index, chunk_text)
                except Exception:
                    logger.exception("Obsidian write error")

    def finalize_document(self, work: DocumentWork):
        """Finalize document processing and update state."""
        self.state_manager.update_document_state(work.doc_id, work.text_hash)

    def process_document(self, document: dict):
        """Process a single document through the complete pipeline."""
        work = self.prepare_document_work(document)
        if not work:
            return

        self.run_downstream_steps(work)
        self.finalize_document(work)

    def run_main_process(self):
        """Run the main document processing loop."""
        try:
            self.bootstrapper.bootstrap_all_services()
            processed = 0

            for document in self.paperless_processor.iter_documents():
                if STOP_EVENT.is_set():
                    logger.info("Shutdown requested during run; stopping early.")
                    break

                try:
                    self.process_document(document)
                    processed += 1
                except Exception as exc:
                    logger.error(f"Failed to process document: {exc}")

            logger.info(f"[run] completed; processed_docs={processed}")

        except Exception as exc:
            logger.critical(f"Fatal error in main process: {exc}")
            raise


class SchedulerCoordinator:
    """Coordinates scheduled execution and handles concurrency."""

    def __init__(self, document_processor: DocumentProcessor):
        self.document_processor = document_processor
        self.run_lock = threading.Lock()
        self.stop_event = threading.Event()

    def run_scheduled_job(self):
        """Execute main processing job if not already running."""
        if self.stop_event.is_set():
            return

        acquired = self.run_lock.acquire(blocking=False)
        if not acquired:
            logger.warning("Previous run still in progress; skipping this schedule tick.")
            return

        try:
            if not self.stop_event.is_set():
                self.document_processor.run_main_process()
        finally:
            self.run_lock.release()

    def run_initial_process(self):
        """Run initial processing if possible."""
        if self.stop_event.is_set():
            logger.info("Shutdown requested before run; skipping main().")
            return

        acquired = self.run_lock.acquire(blocking=False)
        if acquired:
            try:
                self.document_processor.run_main_process()
            finally:
                self.run_lock.release()
        else:
            logger.warning("Importer already running at startup; initial run skipped.")

    def start_scheduler(self):
        """Start the scheduled execution loop."""
        schedule_time = int(os.getenv("SCHEDULE_TIME", "5"))
        schedule.every(schedule_time).minutes.do(self.run_scheduled_job)

        while not self.stop_event.is_set():
            schedule.run_pending()
            time.sleep(1)

    def request_stop(self):
        """Request graceful shutdown."""
        self.stop_event.set()


# Global instances
document_processor = DocumentProcessor()
scheduler_coordinator = SchedulerCoordinator(document_processor)

# Global stop event for signal handling
STOP_EVENT = scheduler_coordinator.stop_event
RUN_LOCK = scheduler_coordinator.run_lock


def main():
    """Main entry point - runs initial processing then starts scheduler."""
    scheduler_coordinator.run_initial_process()
    scheduler_coordinator.start_scheduler()


if __name__ == "__main__":
    LOG_FILE = os.getenv("LOG_FILE", "/data/importer.log")
    try:
        Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    logger.add(
        LOG_FILE,
        rotation="10 MB",
        retention="10 days",
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )

    def handle_signal(signum, frame):
        logger.info(f"Received signal {signum}; requesting shutdown...")
        scheduler_coordinator.request_stop()

    for sig in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
        if sig is not None:
            signal.signal(sig, handle_signal)

    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted.")
        sys.exit(130)
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        logger.info("Importer stopped.")
