import contextlib
import json
from pathlib import Path
from typing import Dict, Any

from loguru import logger
from ..config import Config


class StateManager:
    """Manages processing state and idempotency checks."""

    def __init__(self, state_path: Path | None = None):
        self.state_path = state_path or Config.STATE_PATH
        self.state: Dict[str, Any] = {"last_id": 0, "hashes": {}}
        self._load_state()

    def _load_state(self):
        """Load state from a file if it exists."""
        if self.state_path.exists():
            with contextlib.suppress(Exception):
                self.state = json.loads(self.state_path.read_text(encoding="utf-8"))

    def save_state(self):
        """Save the current state to file."""
        with contextlib.suppress(Exception):
            self.state_path.write_text(json.dumps(self.state), encoding="utf-8")

    def should_process_document(self, doc_id: int) -> bool:
        """Check if a document should be processed based on state."""
        return True if Config.FORCE_REPROCESS else doc_id > self.state.get("last_id", 0)

    def is_document_changed(self, doc_id: int, text_hash: str) -> bool:
        """Check if document content has changed."""
        return (
            True
            if Config.FORCE_REPROCESS
            else text_hash != self.state.get("hashes", {}).get(str(doc_id), "")
        )

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
