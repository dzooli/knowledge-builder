import os
from pathlib import Path
from typing import Optional


class Config:
    """Centralized configuration management for the importer application."""

    # Paperless Configuration
    PAPERLESS_URL: str = os.getenv("PAPERLESS_URL", "http://paperless:8000")
    PAPERLESS_TOKEN: Optional[str] = os.getenv("PAPERLESS_TOKEN")
    PAPERLESS_TOKEN_FILE: Optional[str] = os.getenv("PAPERLESS_TOKEN_FILE")

    # Ollama Configuration
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://ollama:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama31-kb")

    # Neo4j Configuration
    NEO4J_HOST: str = os.getenv("NEO4J_HOST", "host.docker.internal")
    NEO4J_PORT: int = int(os.getenv("NEO4J_PORT", "7687"))
    NEO4J_URL: Optional[str] = os.getenv("NEO4J_URL") or os.getenv("NEO4J_URI")
    NEO4J_USER_ENV: Optional[str] = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER")
    NEO4J_PASS_ENV: Optional[str] = os.getenv("NEO4J_PASSWORD") or os.getenv("NEO4J_PASS")
    NEO4J_DATABASE: Optional[str] = os.getenv("NEO4J_DATABASE")

    # Processing Configuration
    FORCE_REPROCESS: bool = str(os.getenv("FORCE_REPROCESS", "0")).lower() in {"1", "true", "yes"}
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "5000"))
    SCHEDULE_TIME: int = int(os.getenv("SCHEDULE_TIME", "5"))

    # MCP Configuration
    DEFAULT_MCP_CMD = "/app/.venv/bin/mcp-neo4j-memory"
    MEMORY_MCP_CMD: str = os.getenv("MEMORY_MCP_CMD", DEFAULT_MCP_CMD)

    # File Paths
    STATE_PATH: Path = Path(os.getenv("STATE_PATH", "/data/state.json"))
    VAULT_DIR: Path = Path(os.getenv("VAULT_DIR", "/data/obsidian"))
    LOG_FILE: str = os.getenv("LOG_FILE", "/data/importer.log")

    # Logging Controls
    LOG_CHUNK_FULL: bool = str(os.getenv("LOG_CHUNK_FULL", "0")).lower() in {"1", "true", "yes"}
    LOG_CHUNK_PREVIEW_MAX: int = int(os.getenv("LOG_CHUNK_PREVIEW_MAX", "2000"))
    LOG_TOOL_PREVIEW_MAX: int = int(os.getenv("LOG_TOOL_PREVIEW_MAX", "1500"))
    LOG_LLM_OUTPUT_MAX: int = int(os.getenv("LOG_LLM_OUTPUT_MAX", "4000"))

    # Optional Features
    OBSIDIAN_EXPORT: bool = str(os.getenv("OBSIDIAN_EXPORT", "0")).lower() in {"1", "true", "yes"}

    @classmethod
    def initialize_directories(cls):
        """Initialize required directories."""
        cls.VAULT_DIR.mkdir(parents=True, exist_ok=True)


# Initialize directories on import
Config.initialize_directories()
