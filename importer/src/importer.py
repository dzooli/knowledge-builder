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


PAPERLESS_URL: str = os.getenv("PAPERLESS_URL", "http://paperless:8000")
PAPERLESS_TOKEN: Optional[str] = os.getenv("PAPERLESS_TOKEN")
PAPERLESS_TOKEN_FILE: Optional[str] = os.getenv("PAPERLESS_TOKEN_FILE")

OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://ollama:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama31-kb")  # Modelfile-ben temperature=0

NEO4J_HOST: str = os.getenv("NEO4J_HOST", "host.docker.internal")
NEO4J_PORT: int = int(os.getenv("NEO4J_PORT", "7687"))
NEO4J_URL: Optional[str] = os.getenv("NEO4J_URL") or os.getenv("NEO4J_URI")
# Resolve Neo4j credentials for MCP CLI
NEO4J_USER_ENV: Optional[str] = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER")
NEO4J_PASS_ENV: Optional[str] = os.getenv("NEO4J_PASSWORD") or os.getenv("NEO4J_PASS")
NEO4J_DATABASE: Optional[str] = os.getenv("NEO4J_DATABASE")

# Reprocess toggle to bypass id/hash checks for troubleshooting
FORCE_REPROCESS: bool = str(os.getenv("FORCE_REPROCESS", "0")).lower() in {
    "1",
    "true",
    "yes",
}

# Prefer running the MCP server via its console script. It's installed into /app/.venv/bin and on PATH.
DEFAULT_MCP_CMD = "/app/.venv/bin/mcp-neo4j-memory"
FALLBACK_MCP_CMD = "/app/.venv/bin/mcp-neo4j-memory"
MEMORY_MCP_CMD: str = os.getenv("MEMORY_MCP_CMD", DEFAULT_MCP_CMD)

STATE_PATH: Path = Path(os.getenv("STATE_PATH", "/data/state.json"))
VAULT_DIR: Path = Path(os.getenv("VAULT_DIR", "/data/obsidian"))
TZ: str = os.getenv("TZ", "Europe/Budapest")

# Chunk size for splitting text (item 3)
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "5000"))

# Logging controls
LOG_CHUNK_FULL: bool = str(os.getenv("LOG_CHUNK_FULL", "0")).lower() in {"1", "true", "yes"}
LOG_CHUNK_PREVIEW_MAX: int = int(os.getenv("LOG_CHUNK_PREVIEW_MAX", "2000"))
LOG_TOOL_PREVIEW_MAX: int = int(os.getenv("LOG_TOOL_PREVIEW_MAX", "1500"))
LOG_LLM_OUTPUT_MAX: int = int(os.getenv("LOG_LLM_OUTPUT_MAX", "4000"))

VAULT_DIR.mkdir(parents=True, exist_ok=True)

# State
STATE: Dict[str, Any] = {"last_id": 0, "hashes": {}}
if STATE_PATH.exists():
    try:
        STATE = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass


# ====================================
#   Prompt (user-prompt; no system)
# ====================================

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

# Fallback prompt if the agent only read the graph and did not persist any knowledge
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


# ====================================
#   MCP stdio config via adapters
# ====================================

def _neo4j_host_port_from_url(url: str) -> Optional[tuple[str, int]]:
    with contextlib.suppress(Exception):
        p = urlparse(url)
        if p.hostname:
            return p.hostname, p.port or 7687
    return None


def _build_stdio_server_config() -> Dict[str, Any]:
    # Build command and args list for mcp-neo4j-memory
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


async def _load_mcp_tools() -> List[BaseTool]:
    cfg = _build_stdio_server_config()
    client = MultiServerMCPClient(cfg)
    tools = await client.get_tools()
    logger.info(f"[mcp] loaded tools via adapter: {[t.name for t in tools]}")
    # Debug: log required args for each tool to guide prompt shaping
    for t in tools:
        try:
            schema = None
            args_schema = getattr(t, "args_schema", None)
            if args_schema is not None:
                if hasattr(args_schema, "model_json_schema"):
                    schema = args_schema.model_json_schema()
                elif hasattr(args_schema, "schema"):
                    schema = args_schema.schema()
            if isinstance(schema, dict):
                required = schema.get("required") or []
                logger.info(f"[mcp] tool schema name={t.name} required={required}")
        except Exception:
            logger.debug(f"[mcp] tool schema dump failed for {t.name}")
    return tools

# Cache for MCP tools to avoid reloading for each chunk
_MCP_TOOLS_CACHE: List[BaseTool] = []
_MCP_TOOLS_LOADED: bool = False

async def _ensure_mcp_tools() -> List[BaseTool]:
    global _MCP_TOOLS_CACHE, _MCP_TOOLS_LOADED
    if not _MCP_TOOLS_LOADED:
        _MCP_TOOLS_CACHE = await _load_mcp_tools()
        _MCP_TOOLS_LOADED = True
    return _MCP_TOOLS_CACHE


def _tools_by_name_sync() -> Dict[str, BaseTool]:
    try:
        tools = asyncio.run(_ensure_mcp_tools())
    except RuntimeError:
        tools = _MCP_TOOLS_CACHE
    return {t.name: t for t in tools}


def _normalize_params(tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    def norm_name(v: Any) -> Any:
        if isinstance(v, dict) and "name" in v:
            return v.get("name")
        return v

    p = dict(params or {})
    if tool_name == "create_relations":
        rels = p.get("relations")
        if not rels:
            maybe = {k: p.get(k) for k in ("source", "predicate", "relationType", "target", "when", "evidence", "confidence", "sourceId", "chunkId", "sourceUrl") if k in p}
            if any(maybe.values()):
                rels = [maybe]
        out: List[Dict[str, Any]] = []
        for r in rels or []:
            if not isinstance(r, dict):
                continue
            rr = dict(r)
            if "subject" in rr and "source" not in rr:
                rr["source"] = norm_name(rr.pop("subject"))
            if "object" in rr and "target" not in rr:
                rr["target"] = norm_name(rr.pop("object"))
            if "predicate" in rr and "relationType" not in rr:
                rr["relationType"] = rr.pop("predicate")
            rr["source"] = norm_name(rr.get("source"))
            rr["target"] = norm_name(rr.get("target"))
            out.append(rr)
        p = {k: v for k, v in p.items() if k != "relations"}
        p["relations"] = out
    elif tool_name == "add_observations":
        obs = p.get("observations")
        if not obs:
            if "observation" in p and isinstance(p["observation"], dict):
                obs = [p["observation"]]
            elif any(k in p for k in ("entityName", "text", "observations")):
                # Build from flat fields; prefer observations if provided, else wrap text
                observations_list = []
                if isinstance(p.get("observations"), list):
                    observations_list = [str(x) for x in p["observations"]]
                elif "text" in p and p["text"]:
                    observations_list = [str(p["text"])]
                item = {"entityName": p.get("entityName"), "observations": observations_list}
                obs = [item]
        out2: List[Dict[str, Any]] = []
        for o in obs or []:
            if not isinstance(o, dict):
                continue
            oo = dict(o)
            if "entity" in oo and "entityName" not in oo:
                oo["entityName"] = norm_name(oo.pop("entity"))
            if "entity_name" in oo and "entityName" not in oo:
                oo["entityName"] = norm_name(oo.pop("entity_name"))
            if "name" in oo and "entityName" not in oo:
                oo["entityName"] = norm_name(oo.pop("name"))
            # Ensure observations list exists; derive from text if needed
            if "observations" not in oo or not isinstance(oo.get("observations"), list):
                if "text" in oo and oo["text"]:
                    oo["observations"] = [str(oo.pop("text"))]
                else:
                    oo["observations"] = []
            out2.append(oo)
        p = {k: v for k, v in p.items() if k not in {"observations", "observation", "text"}}
        p["observations"] = out2
    elif tool_name == "create_entities":
        ents = p.get("entities")
        if not ents:
            if "entity" in p and isinstance(p["entity"], dict):
                ents = [p["entity"]]
            elif any(k in p for k in ("name", "type")):
                item = {k: p.get(k) for k in ("name", "type") if k in p}
                ents = [item]
            elif "observations" in p and isinstance(p["observations"], list) and p["observations"]:
                first = p["observations"][0]
                if isinstance(first, dict) and "entityName" in first:
                    ents = [{"name": first["entityName"], "type": p.get("type", "Thing")}]
        out3: List[Dict[str, Any]] = []
        for e in ents or []:
            if not isinstance(e, dict):
                continue
            ee = {k: v for k, v in e.items() if k in {"name", "type", "observations"}}
            if "name" in ee:
                if "observations" not in ee or not isinstance(ee.get("observations"), list):
                    ee["observations"] = []
                out3.append(ee)
        p = {k: v for k, v in p.items() if k != "entities"}
        p["entities"] = out3
    return p


def _invoke_tool_by_name(name: str, params: Dict[str, Any]) -> Any:
    tools_by_name = _tools_by_name_sync()
    tool = tools_by_name.get(name)
    if not tool:
        raise RuntimeError(f"Unknown tool: {name}")
    norm = _normalize_params(name, params or {})
    # Try common invocation paths without mutating tool
    # Prefer async if available (StructuredTool often requires ainvoke)
    if hasattr(tool, "ainvoke") and callable(getattr(tool, "ainvoke")):
        return asyncio.run(tool.ainvoke(norm))
    # Then try sync invoke, falling back to async if it complains
    if hasattr(tool, "invoke") and callable(getattr(tool, "invoke")):
        try:
            return tool.invoke(norm)
        except Exception:
            if hasattr(tool, "ainvoke") and callable(getattr(tool, "ainvoke")):
                return asyncio.run(tool.ainvoke(norm))
            raise
    # Legacy run()
    if hasattr(tool, "run") and callable(getattr(tool, "run")):
        try:
            return tool.run(norm)
        except Exception:
            if hasattr(tool, "ainvoke") and callable(getattr(tool, "ainvoke")):
                return asyncio.run(tool.ainvoke(norm))
            raise
    # Callable fallback
    if hasattr(tool, "__call__") and callable(getattr(tool, "__call__")):
        return tool(norm)
    raise RuntimeError(f"Tool {name} is not invokable via known methods")


# ====================================
#   Paperless helpers
# ====================================

def paperless_headers() -> dict:
    token = wait_for_paperless_token()  # infinite wait; set timeout if needed
    return {"Authorization": f"Token {token}"}


def paperless_iter():
    url = f"{PAPERLESS_URL}/api/documents/?ordering=id"
    headers = paperless_headers()
    while url:
        with httpx.Client(timeout=30) as client:
            r = client.get(url, headers=headers)
            r.raise_for_status()
            data = r.json()
            yield from data.get("results", [])
            url = data.get("next")


def paperless_get_document(doc_id: int) -> dict:
    url = f"{PAPERLESS_URL}/api/documents/{doc_id}/"
    headers = paperless_headers()
    with httpx.Client(timeout=30) as client:
        r = client.get(url, headers=headers)
        r.raise_for_status()
        return r.json()


def extract_text(doc: dict) -> str:
    return (doc.get("content") or "").strip()


def chunk_text(t: str, max_chars: int = CHUNK_SIZE) -> List[str]:
    return [t[i:i + max_chars] for i in range(0, len(t), max_chars)] if t else [""]


# ====================================
#   Logging helpers
# ====================================

def _truncate_text(text: Any, limit: int) -> str:
    try:
        s = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)
    except Exception:
        s = str(text)
    if limit <= 0 or len(s) <= limit:
        return s
    tail = len(s) - limit
    return f"{s[:limit]}... [truncated {tail} chars]"


def obsidian_write(doc: dict, idx: int, text: str):
    try:
        slug = f"{doc['id']}_c{idx+1}"
        meta = {
            "title": doc.get("title") or slug,
            "created": doc.get("created"),
            "source": doc.get("download_url"),
            "paperless_id": doc["id"],
            "chunk": idx + 1
        }
        body = "---\n" + json.dumps(meta, ensure_ascii=False, indent=2) + "\n---\n\n" + text + "\n"
        (VAULT_DIR / f"{slug}.md").write_text(body, encoding="utf-8")
        logger.info(f"[obsidian] wrote: {slug}.md ({len(text)} chars)")
    except Exception:
        logger.exception("Obsidian write error")


# ====================================
#   Resource waits
# ====================================

def wait_for_paperless_token(timeout_seconds: int = 0) -> str:
    global PAPERLESS_TOKEN
    if PAPERLESS_TOKEN:
        return PAPERLESS_TOKEN
    token_path = PAPERLESS_TOKEN_FILE
    if not token_path:
        logger.error("Neither PAPERLESS_TOKEN nor PAPERLESS_TOKEN_FILE is specified.")
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
            logger.error("Token not available within the specified time.")
            raise RuntimeError("Token not available within the specified time.")
        time.sleep(2)


def wait_for_neo4j(host: str, port: int, timeout: int = 240):
    logger.info(f"[bootstrap] Waiting for Neo4j at {host}:{port} ...")
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with socket.create_connection((host, port), timeout=2):
                logger.info("[bootstrap] Neo4j available.")
                return
        except OSError:
            time.sleep(2)
    raise RuntimeError("Neo4j not available.")


def wait_for_http(url: str, timeout: int = 240):
    logger.info(f"[bootstrap] Waiting for HTTP service: {url}")
    t0 = time.time()
    while time.time() - t0 < timeout:
        with contextlib.suppress(Exception):
            with httpx.Client(timeout=5) as client:
                r = client.get(url)
                if 200 <= r.status_code < 500:
                    logger.info(f"[bootstrap] Available: {url}")
                    return
        time.sleep(2)
    raise RuntimeError(f"Service not available: {url}")



def _strip_code_fences(s: str) -> str:
    if not s:
        return s
    # Remove opening ```lang (with optional whitespace) and closing ```; support CRLF
    return re.sub(r"```[a-zA-Z]*\s*\r?\n|```", "", s, flags=re.MULTILINE)


def _extract_tool_calls(text: str) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    if not text:
        return calls
    cleaned = _strip_code_fences(text)

    # Try direct JSON parse (array or object)
    with contextlib.suppress(Exception):
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict) and "name" in parsed and "parameters" in parsed:
            return [parsed]
        if isinstance(parsed, list):
            calls = [o for o in parsed if isinstance(o, dict) and "name" in o and "parameters" in o]
            if calls:
                return calls[:10]

    # Regex-based extraction of minimal objects when params are simple (no nested braces)
    pattern = re.compile(r"\{\s*\"name\"\s*:\s*\"[^\"]+\"\s*,\s*\"parameters\"\s*:\s*\{[^{}]*\}\s*\}", re.DOTALL)
    for m in pattern.finditer(cleaned):
        block = m.group(0)
        with contextlib.suppress(Exception):
            obj = json.loads(block)
            if isinstance(obj, dict) and "name" in obj and "parameters" in obj:
                calls.append(obj)
    if calls:
        return calls[:10]

    # Fallback: balanced-brace scanner to extract JSON objects, handling nested braces and strings
    def iter_json_objects(s: str):
        i = 0
        n = len(s)
        while i < n:
            if s[i] == '{':
                start = i
                depth = 0
                i += 1
                in_str = False
                esc = False
                while i < n:
                    c = s[i]
                    if in_str:
                        if esc:
                            esc = False
                        elif c == '\\':
                            esc = True
                        elif c == '"':
                            in_str = False
                    else:
                        if c == '"':
                            in_str = True
                        elif c == '{':
                            depth += 1
                        elif c == '}':
                            if depth == 0:
                                # End of this top-level object
                                yield s[start:i+1]
                                i += 1
                                break
                            depth -= 1
                    i += 1
            else:
                i += 1

    for block in iter_json_objects(cleaned):
        with contextlib.suppress(Exception):
            obj = json.loads(block)
            if isinstance(obj, dict) and "name" in obj and "parameters" in obj:
                calls.append(obj)
    if calls:
        return calls[:10]

    # As a last resort: try to find a top-level JSON array and parse it
    def find_top_level_array(s: str) -> Optional[str]:
        i = 0
        n = len(s)
        while i < n:
            if s[i] == '[':
                start = i
                depth = 0
                i += 1
                in_str = False
                esc = False
                while i < n:
                    c = s[i]
                    if in_str:
                        if esc:
                            esc = False
                        elif c == '\\':
                            esc = True
                        elif c == '"':
                            in_str = False
                    else:
                        if c == '"':
                            in_str = True
                        elif c == '[':
                            depth += 1
                        elif c == ']':
                            if depth == 0:
                                return s[start:i+1]
                            depth -= 1
                    i += 1
            else:
                i += 1
        return None

    with contextlib.suppress(Exception):
        arr_str = find_top_level_array(cleaned)
        if arr_str:
            parsed_arr = json.loads(arr_str)
            if isinstance(parsed_arr, list):
                calls = [o for o in parsed_arr if isinstance(o, dict) and "name" in o and "parameters" in o]
                if calls:
                    return calls[:10]

    return calls[:10]


def _extract_relations_from_calls(calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rels: List[Dict[str, Any]] = []
    for c in calls or []:
        try:
            if c.get("name") != "create_relations":
                continue
            params = c.get("parameters") or {}
            items = params.get("relations") or []
            for r in items:
                if isinstance(r, dict):
                    rels.append(r)
        except Exception:
            continue
    return rels


def _execute_tool_calls(calls: List[Dict[str, Any]]) -> tuple[bool, List[str]]:
    if not calls:
        return False, []
    wrote = False
    touched: List[str] = []
    # Partition calls to ensure entities/observations before relations
    ce_calls = [c for c in calls if c.get("name") == "create_entities"]
    ao_calls = [c for c in calls if c.get("name") == "add_observations"]
    cr_calls = [c for c in calls if c.get("name") == "create_relations"]
    other_calls = [c for c in calls if c.get("name") not in {"create_entities", "add_observations", "create_relations"}]

    ordered = ce_calls + ao_calls + other_calls + cr_calls
    for i, call in enumerate(ordered, start=1):
        name = call.get("name")
        params = call.get("parameters")
        if not isinstance(name, str) or not isinstance(params, dict):
            continue
        try:
            logger.info(f"[agent] executing suggested tool #{i}: {name} args={_truncate_text(params, LOG_TOOL_PREVIEW_MAX)}")
            result = _invoke_tool_by_name(name, params)
            logger.info(f"[agent] suggested tool result #{i} {name}: {_truncate_text(result, LOG_TOOL_PREVIEW_MAX)}")
            if name in {"create_entities", "create_relations", "add_observations"} and not (isinstance(result, str) and str(result).lower().startswith("error")):
                wrote = True
            # Collect touched entities
            if name == "create_entities":
                ents = (params or {}).get("entities") or []
                for e in ents:
                    n = e.get("name") if isinstance(e, dict) else None
                    if isinstance(n, str):
                        touched.append(n)
            elif name == "add_observations":
                obs = (params or {}).get("observations") or []
                for o in obs:
                    n = o.get("entityName") if isinstance(o, dict) else None
                    if isinstance(n, dict):
                        n = n.get("name")
                    if isinstance(n, str):
                        touched.append(n)
            elif name == "create_relations":
                rels = (params or {}).get("relations") or []
                for r in rels:
                    if isinstance(r, dict):
                        s = r.get("source")
                        t = r.get("target")
                        if isinstance(s, dict):
                            s = s.get("name")
                        if isinstance(t, dict):
                            t = t.get("name")
                        if isinstance(s, str):
                            touched.append(s)
                        if isinstance(t, str):
                            touched.append(t)
        except Exception as exc:
            logger.warning(f"[agent] suggested tool error {name}: {exc}")
    # Deduplicate touched
    seen = set()
    touched_unique = [n for n in touched if isinstance(n, str) and n and (n not in seen and not seen.add(n))]
    return wrote, touched_unique


def _extract_entities_from_calls(calls: List[Dict[str, Any]]) -> List[str]:
    names: List[str] = []
    for c in calls or []:
        try:
            name = c.get("name")
            params = c.get("parameters") or {}
            if name == "add_observations":
                obs = params.get("observations") or []
                for o in obs:
                    en = o.get("entityName") if isinstance(o, dict) else None
                    if isinstance(en, dict):
                        en = en.get("name")
                    if isinstance(en, str):
                        names.append(en)
            elif name == "create_entities":
                ents = params.get("entities") or []
                for e in ents:
                    en = e.get("name") if isinstance(e, dict) else None
                    if isinstance(en, str):
                        names.append(en)
            elif name == "create_relations":
                rels = params.get("relations") or []
                for r in rels:
                    if isinstance(r, dict):
                        s = r.get("source")
                        t = r.get("target")
                        if isinstance(s, dict):
                            s = s.get("name")
                        if isinstance(t, dict):
                            t = t.get("name")
                        if isinstance(s, str):
                            names.append(s)
                        if isinstance(t, str):
                            names.append(t)
        except Exception:
            continue
    # Deduplicate while preserving order
    seen = set()
    out: List[str] = []
    for n in names:
        if n not in seen:
            out.append(n)
            seen.add(n)
    return out


def _ensure_evidence_links(entity_names: List[str], source_id: str, chunk_id: str, source_url: str) -> None:
    if not entity_names:
        return
    evidence_name = f"Evidence {source_id}-{chunk_id}"
    try:
        # Upsert Evidence entity
        ce_payload = {"entities": [{
            "name": evidence_name,
            "type": "Evidence",
            "observations": [f"srcId={source_id}", f"chunk={chunk_id}", f"url={source_url or ''}"]
        }]}
        _invoke_tool_by_name("create_entities", ce_payload)
    except Exception as exc:
        logger.warning(f"[agent] evidence entity upsert failed: {exc}")
    # Create evidence relations from all entities to the evidence
    # Exclude self-link if the evidence entity name appears in the list
    sources = [en for en in entity_names if isinstance(en, str) and en and en != evidence_name]
    rels = [{"source": en, "relationType": "evidence", "target": evidence_name} for en in sources]
    if not rels:
        return
    try:
        _invoke_tool_by_name("create_relations", {"relations": rels})
        logger.info(f"[agent] linked {len(rels)} entities to evidence {evidence_name}")
    except Exception as exc:
        logger.warning(f"[agent] evidence relation creation failed: {exc}")



# ====================================
#   Main run for a chunk (with MCP adapters)
# ====================================

def run_for_chunk(source_id: str, chunk_id: str, source_url: str, text: str):
    logger.info(f"[chunk] start doc={source_id} {chunk_id} len={len(text)}")
    if LOG_CHUNK_FULL:
        logger.info(f"[chunk] text doc={source_id} {chunk_id}:\n{text}")
    else:
        logger.info(f"[chunk] preview doc={source_id} {chunk_id}:\n{_truncate_text(text, LOG_CHUNK_PREVIEW_MAX)}")

    try:
        async def _agent_run(prompt_str: str) -> Dict[str, Any]:
            tools = await _ensure_mcp_tools()
            model = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_URL)
            graph = create_react_agent(model, tools)
            logger.info(f"[agent] invoking for doc={source_id} {chunk_id}")
            state: Any = {"messages": prompt_str}
            return await graph.ainvoke(cast(Any, state))

        prompt = (PROMPT_TMPL
                  .replace("{SOURCE_ID}", source_id)
                  .replace("{CHUNK_ID}", chunk_id)
                  .replace("{SOURCE_URL}", source_url or "")
                  .replace("{TEXT}", text))
        result = asyncio.run(_agent_run(prompt))
        msgs = result.get("messages") if isinstance(result, dict) else []
        if msgs:
            last = msgs[-1]
            logger.info(f"[agent] output doc={source_id} {chunk_id}:\n{_truncate_text(getattr(last, 'content', ''), LOG_LLM_OUTPUT_MAX)}")
        wrote = False
        touched: List[str] = []
        relations_to_retry: List[Dict[str, Any]] = []
        # Log tool usage from messages and collect entity names touched
        for i, m in enumerate(msgs or [], start=1):
            try:
                if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                    for tc in m.tool_calls:
                        logger.info(f"[agent] ai#{i} tool_call name={tc.get('name')} args={_truncate_text(tc.get('args'), LOG_TOOL_PREVIEW_MAX)}")
                elif isinstance(m, ToolMessage):
                    logger.info(f"[agent] tool#{i} name={m.name} result={_truncate_text(m.content, LOG_TOOL_PREVIEW_MAX)}")
                    # Count as write only if succeeded (no error in content)
                    if m.name in {"create_entities","create_relations","add_observations","delete_entities","delete_relations","delete_observations"}:
                        content = (m.content or "") if isinstance(m.content, str) else json.dumps(m.content)
                        if not str(content).lower().startswith("error"):
                            wrote = True
                    # Collect touched entities
                    with contextlib.suppress(Exception):
                        raw = m.content
                        data = None
                        if isinstance(raw, str):
                            with contextlib.suppress(Exception):
                                data = json.loads(raw)
                        elif isinstance(raw, (dict, list)):
                            data = raw
                        if m.name == "add_observations" and isinstance(data, list):
                            touched += [d.get("entityName") for d in data if isinstance(d, dict) and isinstance(d.get("entityName"), str)]
                        if m.name == "create_entities" and isinstance(data, list):
                            touched += [d.get("name") for d in data if isinstance(d, dict) and isinstance(d.get("name"), str)]
                        if m.name == "create_relations" and isinstance(data, list):
                            # Server echoes relations; keep for retry later
                            for r in data:
                                if isinstance(r, dict):
                                    relations_to_retry.append(r)
            except Exception:
                logger.info(f"[agent] msg#{i} (unparsable)")

        # If no writes detected, try to parse and execute suggested tool calls from the last AI message
        if not wrote and msgs and isinstance(msgs[-1], AIMessage):
            last_ai = cast(AIMessage, msgs[-1])
            suggested = _extract_tool_calls(getattr(last_ai, "content", ""))
            if suggested:
                # Execute in safe order and collect touched
                exec_wrote, exec_touched = _execute_tool_calls(suggested)
                wrote = exec_wrote or wrote
                touched += exec_touched
                # Also collect relations from suggested calls for retry after upserts
                relations_to_retry += _extract_relations_from_calls(suggested)

        # One-time fallback if still no writes
        if not wrote and text and text.strip():
            logger.warning(f"[agent] no writes detected for doc={source_id} {chunk_id}; retrying with fallback prompt")
            fb_prompt = (PROMPT_TMPL_FALLBACK
                         .replace("{SOURCE_ID}", source_id)
                         .replace("{CHUNK_ID}", chunk_id)
                         .replace("{SOURCE_URL}", source_url or "")
                         .replace("{TEXT}", text))
            result2 = asyncio.run(_agent_run(fb_prompt))
            msgs2 = result2.get("messages") if isinstance(result2, dict) else []
            if msgs2:
                last2 = msgs2[-1]
                logger.info(f"[agent] fallback output doc={source_id} {chunk_id}:\n{_truncate_text(getattr(last2, 'content', ''), LOG_LLM_OUTPUT_MAX)}")
            for i, m in enumerate(msgs2 or [], start=1):
                try:
                    if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                        for tc in m.tool_calls:
                            logger.info(f"[agent] fb-ai#{i} tool_call name={tc.get('name')} args={_truncate_text(tc.get('args'), LOG_TOOL_PREVIEW_MAX)}")
                    elif isinstance(m, ToolMessage):
                        logger.info(f"[agent] fb-tool#{i} name={m.name} result={_truncate_text(m.content, LOG_TOOL_PREVIEW_MAX)}")
                        if m.name in {"create_entities","create_relations","add_observations"}:
                            content = (m.content or "") if isinstance(m.content, str) else json.dumps(m.content)
                            if not str(content).lower().startswith("error"):
                                wrote = True
                        # Collect touched entities from fallback
                        with contextlib.suppress(Exception):
                            raw = m.content
                            data = None
                            if isinstance(raw, str):
                                with contextlib.suppress(Exception):
                                    data = json.loads(raw)
                            elif isinstance(raw, (dict, list)):
                                data = raw
                            if m.name == "add_observations" and isinstance(data, list):
                                touched += [d.get("entityName") for d in data if isinstance(d, dict) and isinstance(d.get("entityName"), str)]
                            if m.name == "create_entities" and isinstance(data, list):
                                touched += [d.get("name") for d in data if isinstance(d, dict) and isinstance(d.get("name"), str)]
                            if m.name == "create_relations" and isinstance(data, list):
                                for r in data:
                                    if isinstance(r, dict):
                                        relations_to_retry.append(r)
                except Exception:
                    logger.info(f"[agent] fb-msg#{i} (unparsable)")
            # Try suggested calls from fallback too
            if not wrote and msgs2 and isinstance(msgs2[-1], AIMessage):
                suggested2 = _extract_tool_calls(getattr(msgs2[-1], "content", ""))
                if suggested2:
                    exec_wrote2, exec_touched2 = _execute_tool_calls(suggested2)
                    wrote = exec_wrote2 or wrote
                    touched += exec_touched2
                    relations_to_retry += _extract_relations_from_calls(suggested2)

        # Final programmatic fallback: ensure at least one write
        if not wrote and text and text.strip():
            try:
                doc_entity = f"Document {source_id}"
                logger.warning(f"[agent] forcing minimal write for doc={source_id} {chunk_id}")
                # create_entities: Document + Evidence
                try:
                    ce_payload = {"entities": [
                        {"name": doc_entity, "type": "Document", "observations": []},
                        {"name": f"Evidence {source_id}-{chunk_id}", "type": "Evidence", "observations": [f"srcId={source_id}", f"chunk={chunk_id}", f"url={source_url or ''}"]}
                    ]}
                    ce_res = _invoke_tool_by_name("create_entities", ce_payload)
                    logger.info(f"[agent] forced create_entities result: {_truncate_text(ce_res, LOG_TOOL_PREVIEW_MAX)}")
                except Exception as exc:
                    logger.warning(f"[agent] forced create_entities failed: {exc}")
                # add_observations (attach short excerpt to Document)
                try:
                    obs_text = text if len(text) <= 1500 else text[:1500]
                    ao_payload = {"observations": [{"entityName": doc_entity, "observations": [obs_text]}]}
                    ao_res = _invoke_tool_by_name("add_observations", ao_payload)
                    logger.info(f"[agent] forced add_observations result: {_truncate_text(ao_res, LOG_TOOL_PREVIEW_MAX)}")
                except Exception as exc:
                    logger.warning(f"[agent] forced add_observations failed: {exc}")
                # evidence relation
                try:
                    _invoke_tool_by_name("create_relations", {"relations": [{
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

        # Ensure evidence links for all entities touched in this chunk
        if touched:
            # Deduplicate
            seen = set()
            touched_unique = [n for n in touched if isinstance(n, str) and n and (n not in seen and not seen.add(n))]
            _ensure_evidence_links(touched_unique, source_id, chunk_id, source_url)

        # Retry domain relations at the very end (after entities/observations and evidence links)
        if relations_to_retry:
            # Deduplicate by (source, relationType/predicate, target)
            dedup_key = set()
            final_relations: List[Dict[str, Any]] = []
            for r in relations_to_retry:
                if not isinstance(r, dict):
                    continue
                s = r.get("source")
                t = r.get("target")
                rt = r.get("relationType") or r.get("predicate")
                # Handle dict endpoints with name field
                if isinstance(s, dict):
                    s = s.get("name")
                if isinstance(t, dict):
                    t = t.get("name")
                key = (s, rt, t)
                if all(isinstance(x, str) and x for x in key) and key not in dedup_key:
                    final_relations.append({"source": s, "relationType": rt, "target": t})
                    dedup_key.add(key)
            if final_relations:
                try:
                    _invoke_tool_by_name("create_relations", {"relations": final_relations})
                    logger.info(f"[agent] retried {len(final_relations)} domain relations at end of chunk")
                except Exception as exc:
                    logger.warning(f"[agent] retrying relations failed: {exc}")

        logger.info(f"[agent] done for doc={source_id} {chunk_id}; messages={len(msgs) if msgs else 0}")
    except Exception as ex:
        logger.error(f"[agent] failure doc={source_id} {chunk_id}: {ex}")
        raise


# ====================================
#   Main – full ETL cycle
# ====================================

def resolve_neo4j_host_port():
    host_port = _neo4j_host_port_from_url(NEO4J_URL) if NEO4J_URL else None
    return host_port or (NEO4J_HOST, NEO4J_PORT)


def bootstrap_services():
    wait_for_http(PAPERLESS_URL, timeout=600)
    nh, np = resolve_neo4j_host_port()
    wait_for_neo4j(nh, np, timeout=600)
    wait_for_http(f"{OLLAMA_URL}/api/tags", timeout=600)
    _ = paperless_headers()
    logger.info("[bootstrap] Services ready.")


def store_state(doc_id):
    STATE["last_id"] = doc_id
    with contextlib.suppress(Exception):
        STATE_PATH.write_text(json.dumps(STATE), encoding="utf-8")
    logger.info(f"[state] advanced last_id={doc_id}")
    return None


def prepare_document_work(doc: dict) -> Optional[DocumentWork]:
    try:
        doc_id = int(doc.get("id", 0))
    except Exception:
        return None

    logger.info(f"[doc] consider id={doc_id} force={FORCE_REPROCESS}")
    if not FORCE_REPROCESS and doc_id <= STATE.get("last_id", 0):
        logger.info(f"[doc] skip id={doc_id} last_id={STATE.get('last_id', 0)}")
        return None

    try:
        detailed = paperless_get_document(doc_id)
    except Exception as exc:
        logger.error(f"[doc] detail fetch failed id={doc_id}: {exc}")
        return None

    text = extract_text(detailed)

    if not text:
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
        return store_state(doc_id)

    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if not FORCE_REPROCESS and STATE["hashes"].get(str(doc_id)) == h:
        logger.info(f"[doc] unchanged hash; skip id={doc_id}")
        return store_state(doc_id)

    chunks = chunk_text(text)
    logger.info(f"[doc] prepared id={doc_id} chunks={len(chunks)} first_len={len(chunks[0]) if chunks else 0}")
    source_url = str(detailed.get("download_url") or "")
    return DocumentWork(doc_id=doc_id, source_url=source_url, chunks=chunks, text_hash=h, doc=detailed)


def run_downstream_steps(work: DocumentWork) -> None:
    for ci, ch in enumerate(work.chunks):
        source_id = str(work.doc_id)
        chunk_id = f"c{ci+1}"
        try:
            run_for_chunk(source_id, chunk_id, work.source_url, ch)
        except Exception as exc:
            logger.warning(f"Agent/MCP error doc {work.doc_id} {chunk_id}: {exc}")

        if str(environ.get("OBSIDIAN_EXPORT", False)).lower() in {"1", "true", "yes"}:
            try:
                obsidian_write(work.doc, ci, ch)
            except Exception:
                logger.exception("Obsidian write error")


def finalize_document(work: DocumentWork) -> None:
    STATE["hashes"][str(work.doc_id)] = work.text_hash
    STATE["last_id"] = work.doc_id
    with contextlib.suppress(Exception):
        STATE_PATH.write_text(json.dumps(STATE), encoding="utf-8")
    logger.info(f"[state] saved id={work.doc_id} hash_prefix={work.text_hash[:8]}")


def process_document(doc):
    work = prepare_document_work(doc)
    if not work:
        return
    run_downstream_steps(work)
    finalize_document(work)


def main():
    if STOP_EVENT.is_set():
        logger.info("Shutdown requested before run; skipping main().")
        return
    try:
        bootstrap_services()
        processed = 0
        for doc in paperless_iter():
            if STOP_EVENT.is_set():
                logger.info("Shutdown requested during run; stopping early.")
                break
            try:
                process_document(doc)
                processed += 1
            except Exception as exc:
                logger.error(f"Failed to process document: {exc}")
        logger.info(f"[run] completed; processed_docs={processed}")
    except Exception as exc:
        logger.critical(f"Fatal error in main: {exc}")
        raise


def schedule_importer():
    schedule_time = int(os.getenv("SCHEDULE_TIME", "5"))

    def _job():
        if STOP_EVENT.is_set():
            return
        acquired = RUN_LOCK.acquire(blocking=False)
        if not acquired:
            logger.warning("Previous run still in progress; skipping this schedule tick.")
            return
        try:
            main()
        finally:
            RUN_LOCK.release()

    schedule.every(schedule_time).minutes.do(_job)

    while not STOP_EVENT.is_set():
        schedule.run_pending()
        time.sleep(1)


# Scheduler coordination primitives
RUN_LOCK = threading.Lock()
STOP_EVENT = threading.Event()


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

    def _handle_signal(signum, frame):
        logger.info(f"Received signal {signum}; requesting shutdown...")
        STOP_EVENT.set()

    for sig in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
        if sig is not None:
            signal.signal(sig, _handle_signal)

    try:
        if RUN_LOCK.acquire(blocking=False):
            try:
                main()
            finally:
                RUN_LOCK.release()
        else:
            logger.warning("Importer already running at startup; initial run skipped.")

        schedule_importer()
    except KeyboardInterrupt:
        logger.info("Interrupted.")
        sys.exit(130)
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        logger.info("Importer stopped.")
