import contextlib
import socket
import time

import httpx
from loguru import logger

from ..config import Config
from ..connectors import Neo4jMemoryConnector, PaperlessConnector


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
        if Config.NEO4J_URL:
            if host_port := Neo4jMemoryConnector.neo4j_host_port_from_url(
                Config.NEO4J_URL
            ):
                return host_port
        return Config.NEO4J_HOST, Config.NEO4J_PORT

    @classmethod
    def bootstrap_all_services(cls):
        """Bootstrap all required services."""
        cls.wait_for_http_service(Config.PAPERLESS_URL, timeout=600)
        host, port = cls.resolve_neo4j_host_port()
        cls.wait_for_neo4j(host, port, timeout=600)
        cls.wait_for_http_service(f"{Config.OLLAMA_URL}/api/tags", timeout=600)
        PaperlessConnector.get_headers()  # Validate token
        logger.info("[bootstrap] Services ready.")
