import contextlib
import json
import re
from typing import Dict, Any, List
from .json_parser import JSONParser
from .text_utils import TextUtils


class ToolCallExtractor:
    """Extracts tool calls from LLM responses using multiple strategies."""

    @classmethod
    def try_direct_json_parse(cls, text: str) -> List[Dict[str, Any]]:
        """Try parsing text as direct JSON (object or array)."""
        with contextlib.suppress(Exception):
            parsed = json.loads(text)

            if JSONParser.is_valid_tool_call(parsed):
                return [parsed]

            if isinstance(parsed, list):
                valid_calls = [obj for obj in parsed if JSONParser.is_valid_tool_call(obj)]
                return valid_calls or []

        return []

    @classmethod
    def try_regex_extraction(cls, text: str) -> List[Dict[str, Any]]:
        """Extract tool calls using regex for simple cases (no nested braces)."""
        pattern = re.compile(
            r"\{\s*\"name\"\s*:\s*\"[^\"]+\"\s*,\s*\"parameters\"\s*:\s*\{[^{}]*\}\s*\}", 
            re.DOTALL
        )

        calls = []
        for match in pattern.finditer(text):
            block = match.group(0)
            with contextlib.suppress(Exception):
                obj = json.loads(block)
                if JSONParser.is_valid_tool_call(obj):
                    calls.append(obj)

        return calls

    @classmethod
    def try_balanced_brace_extraction(cls, text: str) -> List[Dict[str, Any]]:
        """Extract tool calls using balanced brace scanning for nested structures."""
        calls = []
        for block in JSONParser.iter_json_objects(text):
            with contextlib.suppress(Exception):
                obj = json.loads(block)
                if JSONParser.is_valid_tool_call(obj):
                    calls.append(obj)

        return calls

    @classmethod
    def try_array_extraction(cls, text: str) -> List[Dict[str, Any]]:
        """Try to find and parse a top-level JSON array."""
        with contextlib.suppress(Exception):
            if arr_str := JSONParser.find_top_level_array(text):
                parsed_arr = json.loads(arr_str)
                if isinstance(parsed_arr, list):
                    valid_calls = [obj for obj in parsed_arr if JSONParser.is_valid_tool_call(obj)]
                    return valid_calls or []

        return []

    @classmethod
    def extract_tool_calls(cls, text: str) -> List[Dict[str, Any]]:
        """Extract tool calls from the text using multiple parsing strategies."""
        if not text:
            return []

        cleaned = TextUtils.strip_code_fences(text)

        # Try each extraction strategy in order of preference
        strategies = [
            cls.try_direct_json_parse,
            cls.try_regex_extraction,
            cls.try_balanced_brace_extraction,
            cls.try_array_extraction
        ]

        for strategy in strategies:
            if calls := strategy(cleaned):
                return calls[:10]

        return []
