import contextlib
import os
import sys
import json
import time
import socket
import hashlib
import subprocess
import threading
import queue
import signal
from os import environ
from pathlib import Path
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field, ConfigDict
import schedule
from loguru import logger

from langchain.tools import StructuredTool
from langchain.agents import initialize_agent, AgentType
from langchain_community.chat_models import ChatOllama


PAPERLESS_URL: str = os.getenv("PAPERLESS_URL", "http://paperless:8000")
PAPERLESS_TOKEN: Optional[str] = os.getenv("PAPERLESS_TOKEN")
PAPERLESS_TOKEN_FILE: Optional[str] = os.getenv("PAPERLESS_TOKEN_FILE")

OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://ollama:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama31-kb")  # Modelfile-ben temperature=0

NEO4J_HOST: str = os.getenv("NEO4J_HOST", "host.docker.internal")
NEO4J_PORT: int = int(os.getenv("NEO4J_PORT", "7687"))
NEO4J_URL: Optional[str] = os.getenv("NEO4J_URL") or os.getenv("NEO4J_URI")

MEMORY_MCP_CMD: str = os.getenv("MEMORY_MCP_CMD", "mcp-neo4j-memory")

STATE_PATH: Path = Path(os.getenv("STATE_PATH", "/data/state.json"))
VAULT_DIR: Path = Path(os.getenv("VAULT_DIR", "/data/obsidian"))
TZ: str = os.getenv("TZ", "Europe/Budapest")

# Chunk size for splitting text (item 3)
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "5000"))

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
(find_nodes, search_nodes, read_graph, create_entities, delete_entities, create_relations, delete_relations, add_observations, delete_observations).

Rules:
- First SEARCH (find_nodes / search_nodes / query), then UPSERT (delete+recreate), then OBSERVATION (with add_observations), 
  just CREATE when not existing found (create_entities, add_observations and create_relations).
- Every entity/relation/observation must carry: source_id="{SOURCE_ID}", chunk_id="{CHUNK_ID}", source_url="{SOURCE_URL}" (if relevant).
- Do not duplicate: find_nodes → if exists, use it; otherwise create_entities.
- Canonize names: e.g. "OTP Bank Nyrt." → "OTP Bank".
- Date ISO: YYYY-MM-DD (if only year/month known, fill with 01). Money: value (number), unit (USD/HUF/EUR/$/Ft/€).
- Relation examples, you can use other types if needed: offers, announces, acquires, releases, located_in, headquartered_in, founded_on, ceo_of, part_of,
  depends_on, compatible_with, integrates_with, price_of, discounted_to, available_at, published_on, linked_to, cites,
  authored_by, version_of, supersedes, fixes, affected_by, related_to, deprecated_by.
- For every relation/observation add a short evidence snippet and confidence [0.0–1.0].
- Keep calling tools as long as meaningful information can be extracted. Do not write explanations. If nothing more to do: write a single word: DONE.

Useful patterns (guidelines, adjust to the schema expected by the tool):
- create_entities: {"entities":[{"name":"…","type":"Organization","source_id":"{SOURCE_ID}","chunk_id":"{CHUNK_ID}"}]}
- create_relations: {"relations":[{"subject":{"name":"…"},"predicate":"offers","object":{"name":"…"},"when":"2025-08-01","evidence":"…","confidence":0.8,"source_id":"{SOURCE_ID}","chunk_id":"{CHUNK_ID}"}]}
- add_observations: {"observations":[{"entity":{"name":"…"},"text":"…","source_url":"{SOURCE_URL}","confidence":0.6,"source_id":"{SOURCE_ID}","chunk_id":"{CHUNK_ID}"}]}

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
#   LangChain Tool wrap – FULL toolset
# ====================================

class NodeQueryArgs(BaseModel):
    query: dict = Field(..., description="Search conditions for find_nodes/search_nodes/read_graph.")


class EntitiesArgs(BaseModel):
    entities: list = Field(..., description="List of entities for create_entities/delete_entities.")


class RelationsArgs(BaseModel):
    relations: list = Field(..., description="List of relations for create_relations/delete_relations.")


class ObservationsArgs(BaseModel):
    observations: list = Field(..., description="List of observations for add_observations/delete_observations.")


# ====================================
#   MCP STDIO client – JSON-RPC/LSP
# ====================================

class MCPClient:
    """
    STDIO MCP client for neo4j-contrib 'mcp-neo4j-memory' server.
    Public:
      - initialize()
      - tools_list()
      - call_tool(name, arguments)
      - close()
    """
    def __init__(self, cmd: str):
        self.proc = subprocess.Popen(
            cmd,
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self._q: "queue.Queue[dict]" = queue.Queue()
        self._id = 0
        self._alive = True
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self):
        f = self.proc.stdout
        while self._alive:
            header = b""
            # Read LSP headers
            while b"\r\n\r\n" not in header:
                line = f.readline()
                if not line:
                    self._alive = False
                    return
                header += line
            head, _, _ = header.partition(b"\r\n\r\n")
            length = 0
            for ln in head.split(b"\r\n"):
                if ln.lower().startswith(b"content-length:"):
                    try:
                        length = int(ln.split(b":", 1)[1].strip())
                    except Exception:
                        length = 0
            body = f.read(length)
            with contextlib.suppress(Exception):
                msg = json.loads(body.decode("utf-8"))
                self._q.put(msg)

    def _send(self, obj: dict):
        raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        header = f"Content-Length: {len(raw)}\r\n\r\n".encode("utf-8")
        self.proc.stdin.write(header + raw)
        self.proc.stdin.flush()

    def _request(self, method: str, params: Optional[dict] = None, timeout: int = 60) -> dict:
        self._id += 1
        rid = self._id
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                msg = self._q.get(timeout=timeout)
            except queue.Empty:
                break
            if msg.get("id") == rid:
                if "error" in msg:
                    raise RuntimeError(msg["error"])
                return msg.get("result")
        raise TimeoutError(f"MCP request timeout: {method}")

    def initialize(self):
        return self._request("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}}, timeout=30)

    def tools_list(self) -> dict:
        return self._request("tools/list", {}, timeout=30)

    def call_tool(self, name: str, arguments: dict, timeout: int = 120) -> dict:
        return self._request("tools/call", {"name": name, "arguments": arguments}, timeout=timeout)

    def close(self):
        with contextlib.suppress(Exception):
            self._alive = False
            if self.proc and self.proc.poll() is None:
                self.proc.terminate()


def _neo4j_host_port_from_url(url: str) -> Optional[tuple[str, int]]:
    """
    Parse a Neo4j connection URL and extract the host and port.

    Args:
        url: The Neo4j connection URL.

    Returns:
        A tuple containing the host and port if available, otherwise None.
    """
    with contextlib.suppress(Exception):
        p = urlparse(url)
        if p.hostname:
            return p.hostname, p.port or 7687
    return None


# ====================================
#   Helper: wait for resources
# ====================================

def wait_for_paperless_token(timeout_seconds: int = 0) -> str:
    """
    Read token from environment variable or file.
    If timeout_seconds==0, wait indefinitely until the file contains a valid (not 'PENDING') token.
    """
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
    """
    Waits for a Neo4j database to become available at the specified host and port within a given
    timeout. Continuously checks connectivity until either a successful connection is made
    or the timeout period has elapsed. Logs the status during the wait process.

    :param host: The hostname or IP address of Neo4j to connect to.
    :type host: str
    :param port: The port number on which Neo4j is listening.
    :type port: int
    :param timeout: The maximum time, in seconds, to wait for Neo4j to become available. Default
                    is 240 seconds.
    :type timeout: int, optional
    :raises RuntimeError: If the Neo4j instance is not available within the timeout period.
    :return: None
    """
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
    """
    Waits for an HTTP service at the given URL to become available, checking periodically
    within a specified timeout period. The status of the HTTP service is considered
    available if the server responds with a status code of 200.

    This function will suppress errors during HTTP requests and retry every two seconds
    until the service becomes available or the timeout is exceeded. If the timeout
    is exceeded, a RuntimeError is raised indicating that the service is not available.

    :param url: str
        The target URL of the HTTP service to check for availability.
    :param timeout: int
        The maximum amount of time in seconds to wait for the HTTP service to become available.
        Defaults to 240 seconds.
    :return: None
        The function returns None if the service is successfully detected as available within
        the timeout period.
    :raises RuntimeError:
        Raised if the service does not become available within the specified timeout period.
    """
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


def extract_text(doc: dict) -> str:
    return (doc.get("content") or "").strip()


def chunk_text(t: str, max_chars: int = CHUNK_SIZE) -> List[str]:
    return [t[i:i + max_chars] for i in range(0, len(t), max_chars)] if t else [""]


def obsidian_write(doc: dict, idx: int, text: str):
    """
    Writes a Markdown file formatted for Obsidian, including metadata and content,
    to a specified directory. Metadata includes information such as the document
    title, creation date, source URL, unique identifier, and chunk index.

    :param doc: A dictionary representing the document metadata. Expected keys are:
        - id (str): Unique identifier for the document.
        - title (str, optional): Title of the document.
        - created (str, optional): Date when the document was created.
        - download_url (str, optional): URL to the original document source.
    :param idx: The index of the current chunk in the document. Used to create
        a unique slug and filename.
    :param text: The content of the chunk to be written into the Markdown file.
    :return: None
    """
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
    except Exception:
        logger.exception("Obsidian write error")


def build_tools(mcp: MCPClient) -> List[StructuredTool]:
    """
    Builds and returns a list of structured tools based on MCP tools available.

    This function creates a collection of MCP tools by querying the MCP server for
    available tools. Tools that are verified to exist on the MCP server are wrapped
    with functionality to handle input and communicate with the MCP server for their
    respective purposes, such as searching nodes, managing entities, and adding or
    deleting observations.

    :param mcp: An instance of the MCPClient used to communicate with the MCP server.
    :return: A list of StructuredTool objects, each representing a specific MCP tool.
    :rtype: List[StructuredTool]
    """
    advertised = {t["name"] for t in mcp.tools_list().get("tools", [])}
    tools: List[StructuredTool] = []

    def ensure(name: str):
        if name not in advertised:
            raise RuntimeError(f"The '{name}' tool is not available on the MCP server. Advertised: {sorted(advertised)}")

    def wrap(name: str, schema):
        ensure(name)

        def _f(args):
            payload = args.dict() if hasattr(args, "dict") else (args or {})
            return mcp.call_tool(name, payload)
        return StructuredTool.from_function(name=name, description=f"MCP tool: {name}", func=_f, args_schema=schema)

    # Searchers
    tools.append(wrap("find_nodes", NodeQueryArgs))
    tools.append(wrap("search_nodes", NodeQueryArgs))

    # read_graph – usually without parameters
    if "read_graph" in advertised:
        def _read_graph(_: dict = None):
            return mcp.call_tool("read_graph", {})
        tools.append(StructuredTool.from_function(name="read_graph", description="MCP tool: read_graph", func=_read_graph))

    # CRUD
    tools.append(wrap("create_entities", EntitiesArgs))
    tools.append(wrap("delete_entities", EntitiesArgs))
    tools.append(wrap("create_relations", RelationsArgs))
    tools.append(wrap("delete_relations", RelationsArgs))
    tools.append(wrap("add_observations", ObservationsArgs))
    tools.append(wrap("delete_observations", ObservationsArgs))

    return tools


# ====================================
#   Main run for a chunk
# ====================================

def run_for_chunk(mcp_cmd: str, source_id: str, chunk_id: str, source_url: str, text: str):
    mcp = MCPClient(mcp_cmd)
    try:
        mcp.initialize()
        tools = build_tools(mcp)
        llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_URL)  # temp=0 a modellprofilban

        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            max_iterations=25,
            handle_parsing_errors=True,
        )

        prompt = (PROMPT_TMPL
                  .replace("{SOURCE_ID}", source_id)
                  .replace("{CHUNK_ID}", chunk_id)
                  .replace("{SOURCE_URL}", source_url or "")
                  .replace("{TEXT}", text))

        _ = agent.invoke({"input": prompt})
    finally:
        mcp.close()


# ====================================
#   Main – full ETL cycle
# ====================================

def resolve_neo4j_host_port():
    """
    Resolve Neo4j host and port from the environment variables or defaults.
    """
    host_port = _neo4j_host_port_from_url(NEO4J_URL) if NEO4J_URL else None
    return host_port or (NEO4J_HOST, NEO4J_PORT)


def bootstrap_services():
    """
    Bootstrap and wait for required services: Paperless, Neo4j, and Ollama.
    """
    wait_for_http(PAPERLESS_URL, timeout=600)
    nh, np = resolve_neo4j_host_port()
    wait_for_neo4j(nh, np, timeout=600)
    wait_for_http(f"{OLLAMA_URL}/api/tags", timeout=600)
    _ = paperless_headers()


def _extracted_from_prepare_document_work(doc_id):
    STATE["last_id"] = doc_id
    with contextlib.suppress(Exception):
        STATE_PATH.write_text(json.dumps(STATE), encoding="utf-8")
    return None


def prepare_document_work(doc: dict) -> Optional[DocumentWork]:
    """Prepare a document processing unit or return None if it should be skipped.
    - Skips when id <= last_id
    - Skips and advances state when no text
    - Skips and advances state when hash unchanged
    """
    try:
        doc_id = int(doc.get("id", 0))
    except Exception:
        return None

    if doc_id <= STATE.get("last_id", 0):
        return None

    text = extract_text(doc)

    if not text:
        return _extracted_from_prepare_document_work(doc_id)
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if STATE["hashes"].get(str(doc_id)) == h:
        return _extracted_from_prepare_document_work(doc_id)

    chunks = chunk_text(text)
    source_url = str(doc.get("download_url") or "")
    return DocumentWork(doc_id=doc_id, source_url=source_url, chunks=chunks, text_hash=h, doc=doc)


def run_downstream_steps(work: DocumentWork) -> None:
    """Run agent per chunk and optional Obsidian export."""
    for ci, ch in enumerate(work.chunks):
        source_id = str(work.doc_id)
        chunk_id = f"c{ci+1}"
        try:
            run_for_chunk(MEMORY_MCP_CMD, source_id, chunk_id, work.source_url, ch)
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


def process_document(doc):
    """Orchestrate document processing with clear SRP-compliant steps."""
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
        for doc in paperless_iter():
            if STOP_EVENT.is_set():
                logger.info("Shutdown requested during run; stopping early.")
                break
            try:
                process_document(doc)
            except Exception as exc:
                logger.error(f"Failed to process document: {exc}")
    except Exception as exc:
        logger.critical(f"Fatal error in main: {exc}")
        raise


def schedule_importer():
    """
    Schedule the main function to run periodically based on SCHEDULE_TIME (default: 5 minutes).
    Prevent overlapping runs and support graceful shutdown.
    """
    schedule_time = int(os.getenv("SCHEDULE_TIME", "5"))  # Time in minutes

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
    # Configure loguru with a rotating file handler
    LOG_FILE = os.getenv("LOG_FILE", "/data/importer.log")
    try:
        Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    logger.add(
        LOG_FILE,
        rotation="10 MB",  # Rotate a log file after it reaches 10 MB
        retention="10 days",  # Keep logs for 10 days
        enqueue=True,  # Thread-safe logging
        backtrace=True,  # Include backtrace in errors
        diagnose=True,  # Include detailed diagnostics
    )

    # Signal handlers for graceful shutdown
    def _handle_signal(signum, frame):
        logger.info(f"Received signal {signum}; requesting shutdown...")
        STOP_EVENT.set()

    for sig in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
        if sig is not None:
            signal.signal(sig, _handle_signal)

    try:
        # Run the first import immediately without overlap
        if RUN_LOCK.acquire(blocking=False):
            try:
                main()
            finally:
                RUN_LOCK.release()
        else:
            logger.warning("Importer already running at startup; initial run skipped.")

        # Start the scheduler loop (blocks until STOP_EVENT)
        schedule_importer()
    except KeyboardInterrupt:
        logger.info("Interrupted.")
        sys.exit(130)
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        # Final cleanup hook if needed in future (e.g., close pools)
        logger.info("Importer stopped.")
