"""High-performance JSON and text extraction utilities using orjson.

This module consolidates the various ad-hoc JSON extraction strategies into
one small set of functions:

- safe_loads: attempt fast parse with orjson, fallback to stdlib json.
- extract_first_json: locate the first plausible JSON object or array in text
  (optionally inside code fences) and parse it.

The goal is to reduce complexity while retaining robustness for model output
that may include code fences, prefixes (like JSONSTART) or surrounding prose.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable, Union, cast
import re
import json as _json

try:  # pragma: no cover - import guard
    import orjson  # type: ignore
except ImportError:  # pragma: no cover - orjson not installed
    orjson = None  # type: ignore

_JSON_BLOCK_RE = re.compile(r"```[a-zA-Z0-9]*\n(.*?)```", re.DOTALL)


@runtime_checkable
class _JSONLoads(Protocol):
    def __call__(
        self, __data: Union[str, bytes]
    ) -> Any:  # pragma: no cover - protocol definition
        ...


def _get_orjson_loader() -> Optional[_JSONLoads]:
    """Return orjson.loads if available and well-typed, else None."""
    if orjson is None:  # pragma: no cover - runtime branch when not installed
        return None
    loads = getattr(orjson, "loads", None)
    if callable(loads):
        return cast(_JSONLoads, loads)
    return None


def safe_loads(data: str | bytes) -> Any:
    """Parse JSON with orjson if available, else stdlib json.

    This wrapper avoids mypy/pyright complaints about dynamic attribute access
    while keeping a tiny, fast happy path.
    """
    loader = _get_orjson_loader()
    if loader is not None:  # pragma: no branch - trivial branch
        return loader(data)
    if isinstance(data, bytes):
        data = data.decode("utf-8", errors="replace")
    return _json.loads(data)


def _strip_code_fences(text: str) -> str:
    # Prefer the largest JSON-looking block inside fences if multiple.
    blocks = _JSON_BLOCK_RE.findall(text)
    if blocks:
        # Choose block that starts with {{ or [ after trimming whitespace.
        for blk in blocks:
            s = blk.strip()
            if s.startswith("{") or s.startswith("["):
                return s
        return blocks[0].strip()
    return text


def _normalize_quotes(text: str) -> str:
    return text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")


def find_json_substring(text: str) -> Optional[str]:
    """Find first balanced JSON object or array substring.

    Light-weight single pass using stack accounting for strings/escapes.
    Returns substring or None.
    """
    starts = ["[", "{"]
    stack: list[str] = []
    in_string = False
    escaped = False
    start_index: Optional[int] = None

    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch in starts and not stack:
            start_index = i
            stack.append(ch)
            continue
        if ch in "[{" and stack:
            stack.append(ch)
            continue
        if ch in "]}" and stack:
            stack.pop()
            # No strict type pairing enforcement for robustness; optional check:
            # if (opener == '{' and ch != '}') or (opener == '[' and ch != ']'): continue
            if not stack and start_index is not None:
                # Return inclusive slice
                return text[start_index : i + 1]
    return None


def extract_first_json(text: str) -> Optional[Any]:
    """Extract and parse the first JSON object/array from model output.

    Steps:
      1. Normalize smart quotes.
      2. If marker 'JSONSTART' present, keep only trailing part.
      3. Strip code fences to get inner block.
      4. Quick check: direct parse if startswith { or [.
      5. Fallback: scan for first balanced substring.

    Returns parsed object or None if not found/parse error.
    """
    if not text:
        return None
    text = _normalize_quotes(text)
    marker = "JSONSTART"
    if marker in text:
        text = text.split(marker)[-1]
    stripped = _strip_code_fences(text).strip()
    candidates = []
    if stripped.startswith("{") or stripped.startswith("["):
        candidates.append(stripped)
    else:
        if sub := find_json_substring(stripped):
            candidates.append(sub)
    for cand in candidates:
        try:
            return safe_loads(cand)
        except (ValueError, TypeError):  # pragma: no cover - continue other candidates
            continue
    return None


__all__ = ["safe_loads", "extract_first_json", "find_json_substring"]
