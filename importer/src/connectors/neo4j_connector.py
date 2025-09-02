import asyncio
import contextlib
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from loguru import logger

from config import Config
from utils import ToolCallNormalizer


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
        base_cmd = Config.MEMORY_MCP_CMD or Config.DEFAULT_MCP_CMD
        db_url = Config.NEO4J_URL or f"bolt://{Config.NEO4J_HOST}:{Config.NEO4J_PORT}"
        args: List[str] = ["--db-url", db_url]

        if Config.NEO4J_USER_ENV:
            args += ["--username", Config.NEO4J_USER_ENV]
        if Config.NEO4J_PASS_ENV:
            args += ["--password", Config.NEO4J_PASS_ENV]
        if Config.NEO4J_DATABASE:
            args += ["--database", Config.NEO4J_DATABASE]

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
        """Get the tool dictionary by name (synchronous)."""
        try:
            tools = asyncio.run(self.ensure_mcp_tools())
        except RuntimeError:
            tools = self._tools_cache
        return {t.name: t for t in tools}

    @staticmethod
    def try_async_invoke(tool: BaseTool, params: Dict[str, Any]) -> Any:
        """Try to invoke the tool using the async ainvoke method."""
        if not hasattr(tool, 'ainvoke'):
            return None

        try:
            return asyncio.run(tool.ainvoke(params))
        except Exception:
            return None

    @staticmethod
    def try_sync_invoke(tool: BaseTool, params: Dict[str, Any]) -> Any:
        """Try to invoke the tool using the sync invoke method."""
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
        """Try to invoke the tool using the legacy run method."""
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
