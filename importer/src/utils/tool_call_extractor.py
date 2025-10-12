from typing import Dict, Any, List
from .json_utils import extract_first_json


class ToolCallExtractor:  # pragma: no cover - thin orchestration
    """Simplified extractor using high-performance json_utils.

    We expect model output to contain either a JSON array or object describing
    tool calls. Only a single pass extraction is attempted to reduce
    complexity and maintenance burden.
    """

    @staticmethod
    def _is_valid_tool_call(obj: Any) -> bool:  # noqa: ANN401 - Any is intentional
        return isinstance(obj, dict) and "name" in obj and "parameters" in obj

    @classmethod
    def extract_tool_calls(cls, text: str) -> List[Dict[str, Any]]:
        if not text:
            return []
        parsed = extract_first_json(text)
        if parsed is None:
            return []
        items = parsed if isinstance(parsed, list) else [parsed]
        valid = [i for i in items if cls._is_valid_tool_call(i)]
        return valid[:10]
