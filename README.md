# Knowledge Builder

## Description

This project is an **automated ETL pipeline**: knowledge is extracted from documents OCR-ed by Paperless-ngx using the *
*Ollama** LLM, then loaded into a **Neo4j** graph via the official **Neo4j Memory MCP** server. The loading is performed
by a **LangChain Agent**; tool calls are **delegated to the LLM itself**. Optionally, the raw text can also be exported
to an **Obsidian** vault. The importer now includes a **scheduler** for periodic execution and **verbose logging** using
`loguru`.

## ✨ Components

- **Paperless-ngx** – OCR and storage for screenshots/documents
- **Ollama** – local LLM (`llama31-kb`, *temperature=0* profile)
- **Importer (LangChain Agent)** – Paperless → chunk → prompt → LLM → Memory MCP tool calls (STDIO) → Neo4j
- **Neo4j** – graph database + web UI (Browser)
- **Memory MCP server** – `mcp-neo4j-memory` (STDIO, full toolset)
- **Scheduler** – Executes the importer periodically (default: every 5 minutes)
- **Loguru Logging** – Thread-safe, rotating logs for better diagnostics

## 📂 Directory Structure

    neo4j-stack/
      docker-compose.yml    # Neo4j separate Compose
    paperless/             # Paperless-ngx
        data/              # Paperless data
        media/             # Paperless media
    importer/
      src/                 # Python source code
        importer.py        # final Python script
      Dockerfile
    modelfile/Modelfile    # Ollama model profile (temperature=0)
    bootstrap/             # Paperless token goes here
    data/                  # state (state.json), Obsidian export
    docker-compose.yml     # Paperless, Ollama, Importer

## ✅ Prerequisites

- Docker + Docker Compose
- Free ports: `7474`, `7687`, `8900`, `11434`
- On Linux, Compose already includes: `extra_hosts: host.docker.internal:host-gateway`

## 🚀 Quickstart

1) **Start Neo4j (separate Compose)**
   ```bash
   cd neo4j-stack
   docker compose up -d # Browser: http://localhost:7474 (user: neo4j, pass: testpass, set in neo4j-stack compose)
   ```

2) **Start KB stack (Paperless, Ollama, Importer)**
   ```bash
   cd <PROJECT_ROOT>
   docker compose up -d --build
   ```

3) **Paperless token**  
   On first start, `paperless-token-init` tries to request an API token. If it fails, it creates:
   ```bash
    ./bootstrap/paperless_token.txt  # content: PENDING
   ```
   Open the Paperless UI (http://localhost:8900) → *My Profile → Generate token* and enter it in the file. The *
   *Importer** will detect it within 5 seconds and start.

## 🔧 Configuration (key envs)

- **Paperless**

      PAPERLESS_ADMIN_USER=admin
      PAPERLESS_ADMIN_PASSWORD=adminpass # (first start)
      PAPERLESS_URL=http://paperless:8000
      PAPERLESS_TOKEN_FILE=/bootstrap/paperless_token.txt

- **Neo4j (runs on host)**

      NEO4J_URL=bolt://host.docker.internal:7687
      NEO4J_USERNAME=neo4j
      NEO4J_PASSWORD=testpass
      (duplicate variables also provided for compatibility: `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASS`)

- **Ollama**
    - Modelfile:

          FROM llama3.1:8b
          PARAMETER temperature 0

    - The `ollama-model-init` container handles pull + create, profile name: **llama31-kb**

- **Importer**

      MEMORY_MCP_CMD=mcp-neo4j-memory #  (STDIO default)
      STATE_PATH=/data/state.json
      VAULT_DIR=/data/obsidian
      OBSIDIAN_EXPORT=true            # Optional if you want Obsidian export
      SCHEDULE_TIME=5                 # Optional, schedule interval in minutes

- **Importer / pyproject.toml dependencies** (add if missing)

      requests
      pydantic
      langchain
      langchain-community
      mcp-neo4j-memory
      schedule
      loguru

## 🧠 How the Importer Works

1. Paperless API: list new/modified documents
2. Extract text, chunk into 5000 characters
3. **User-prompt** (no system-prompt) → Ollama (`llama31-kb`)
4. The LLM **directly** calls Memory MCP tools (LangChain StructuredTool wrap):
    - `find_nodes`, `search_nodes`, `read_graph`
    - `create_entities`, `delete_entities`
    - `create_relations`, `delete_relations`
    - `add_observations`, `delete_observations`
5. For every created item, pass: `source_id`, `chunk_id`, `source_url`
6. Optional: Obsidian export if enabled (`data/obsidian/`)
7. **Scheduler**: Runs the importer periodically (default: every 5 minutes)
8. **Logging**: Logs are written to `importer.log` with rotation after 10 MB.

## 🔍 Testing

- **Logs**:

      docker compose logs -f importer

  On startup you will see: *Neo4j available*, MCP *tools/list*, then the ReAct agent steps and tool calls.


- **MCP client binary** in the container:

      docker exec -it importer bash
      which mcp-neo4j-memory
      NEO4J_URL="bolt://host.docker.internal:7687" \
      NEO4J_USERNAME="neo4j" \
      NEO4J_PASSWORD="testpass" \
      mcp-neo4j-memory

## 🧯 Troubleshooting

- **`paperless_token.txt` = PENDING**  
  Generate a new token in the Paperless UI and write it to the file. The Importer will automatically proceed.

- **Neo4j not available**  
  Check if `neo4j-stack` is running. On Linux, `extra_hosts: host.docker.internal:host-gateway` is important.

- **Ollama model not created**  
  Check the `ollama-model-init` logs; the profile name is `llama31-kb`.

- **Tool errors / missing tools**  
  The log line *Available MCP tools* shows the tools advertised by the server. If there is a version mismatch, update
  the `mcp-neo4j-memory` package.

## 📜 License

MIT
