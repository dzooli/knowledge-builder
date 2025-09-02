import json
import re
from typing import Any, List
from config import Config


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
    def chunk_text(text: str, max_chars: int = None) -> List[str]:
        """Split text into chunks of a specified size."""
        if max_chars is None:
            max_chars = Config.CHUNK_SIZE
        return [text[i:i + max_chars] for i in range(0, len(text), max_chars)] if text else [""]
