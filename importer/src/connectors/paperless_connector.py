import contextlib
import json
import os
import time
from pathlib import Path
from typing import Dict, Iterator

import httpx
from loguru import logger

from ..config import Config


class PaperlessConnector:
    """Handles document retrieval and processing from Paperless."""

    @staticmethod
    def wait_for_token(timeout_seconds: int = 0) -> str:
        """Wait for a Paperless authentication token."""
        if Config.PAPERLESS_TOKEN:
            return Config.PAPERLESS_TOKEN

        token_path = Config.PAPERLESS_TOKEN_FILE
        if not token_path:
            raise RuntimeError(
                "Neither PAPERLESS_TOKEN nor PAPERLESS_TOKEN_FILE is specified."
            )

        logger.info(f"[bootstrap] Watching token: {token_path}")
        deadline = (time.time() + timeout_seconds) if timeout_seconds > 0 else None

        while True:
            with contextlib.suppress(Exception):
                if os.path.isfile(token_path) and os.path.getsize(token_path) > 0:
                    content = Path(token_path).read_text(encoding="utf-8").strip()
                    if content and content.upper() != "PENDING":
                        logger.info("[bootstrap] Paperless token read.")
                        return content

            if deadline and time.time() > deadline:
                raise RuntimeError("Token not available within the specified time.")
            time.sleep(2)

    @classmethod
    def get_headers(cls) -> Dict[str, str]:
        """Get authorization headers for Paperless API."""
        token = cls.wait_for_token()
        return {"Authorization": f"Token {token}"}

    @classmethod
    def iter_documents(cls) -> Iterator[Dict]:
        """Iterate through all documents from Paperless API."""
        url = f"{Config.PAPERLESS_URL}/api/documents/?ordering=id"
        headers = cls.get_headers()

        while url:
            with httpx.Client(timeout=30) as client:
                response = client.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()
                yield from data.get("results", [])
                url = data.get("next")

    @classmethod
    def get_document(cls, doc_id: int) -> Dict:
        """Get detailed document data by ID."""
        url = f"{Config.PAPERLESS_URL}/api/documents/{doc_id}/"
        headers = cls.get_headers()

        with httpx.Client(timeout=30) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()
            return response.json()

    @staticmethod
    def extract_text(document: Dict) -> str:
        """Extract text content from document."""
        return (document.get("content") or "").strip()

    @staticmethod
    def write_obsidian_note(document: Dict, chunk_index: int, text: str):
        """Write chunk to Obsidian vault (optional)."""
        try:
            slug = f"{document['id']}_c{chunk_index + 1}"
            meta = {
                "title": document.get("title") or slug,
                "created": document.get("created"),
                "source": document.get("download_url"),
                "paperless_id": document["id"],
                "chunk": chunk_index + 1,
            }
            body = (
                "---\n"
                + json.dumps(meta, ensure_ascii=False, indent=2)
                + "\n---\n\n"
                + text
                + "\n"
            )
            (Config.VAULT_DIR / f"{slug}.md").write_text(body, encoding="utf-8")
            logger.info(f"[obsidian] wrote: {slug}.md ({len(text)} chars)")
        except Exception:
            logger.exception("Obsidian write error")
