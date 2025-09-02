from typing import Any, Optional, Iterator


class JSONParser:
    """JSON parsing utilities for extracting structured data."""

    @staticmethod
    def is_valid_tool_call(obj: Any) -> bool:
        """Check if an object is a valid tool call with name and parameters."""
        return isinstance(obj, dict) and "name" in obj and "parameters" in obj

    @staticmethod
    def extract_json_object_at_position(text: str, start_pos: int) -> Optional[tuple[str, int]]:
        """Extract a complete JSON object starting at the given position."""
        depth = 0
        i = start_pos + 1  # Skip opening brace
        in_string = False
        escaped = False

        while i < len(text):
            char = text[i]

            if in_string:
                if escaped:
                    escaped = False
                elif char == '\\':
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '{':
                depth += 1
            elif char == '}':
                if depth == 0:
                    return text[start_pos:i+1], i
                depth -= 1
            elif char == '"':
                in_string = True
            i += 1

        return None

    @staticmethod
    def extract_json_array_at_position(text: str, start_pos: int) -> Optional[str]:
        """Extract a complete JSON array starting at the given position."""
        depth = 0
        i = start_pos + 1  # Skip opening bracket
        in_string = False
        escaped = False

        while i < len(text):
            char = text[i]

            if in_string:
                if escaped:
                    escaped = False
                elif char == '\\':
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '[':
                depth += 1
            elif char == ']':
                if depth == 0:
                    return text[start_pos:i+1]
                depth -= 1
            elif char == '"':
                in_string = True
            i += 1

        return None

    @classmethod
    def iter_json_objects(cls, text: str) -> Iterator[str]:
        """Yield JSON objects from string using balanced brace scanning."""
        i = 0
        n = len(text)

        while i < n:
            if text[i] == '{':
                if json_obj := cls.extract_json_object_at_position(text, i):
                    yield json_obj[0]
                    i = json_obj[1] + 1
                else:
                    i += 1
            else:
                i += 1

    @classmethod
    def find_top_level_array(cls, text: str) -> Optional[str]:
        """Find and extract the first top-level JSON array in the string."""
        for i in range(len(text)):
            if text[i] == '[':
                if array_content := cls.extract_json_array_at_position(text, i):
                    return array_content
        return None
