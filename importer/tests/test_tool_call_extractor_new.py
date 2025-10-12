from importer.src.utils.tool_call_extractor import ToolCallExtractor


def test_tool_call_extractor_direct_array():
    text = '[{"name": "create_entities", "parameters": {"entities": []}}, {"name": "add_observations", "parameters": {"observations": []}}]'
    calls = ToolCallExtractor.extract_tool_calls(text)
    assert len(calls) == 2
    assert {c["name"] for c in calls} == {"create_entities", "add_observations"}


def test_tool_call_extractor_code_fence():
    text = """```json
[{"name":"create_relations", "parameters": {"relations": []}}]
```"""
    calls = ToolCallExtractor.extract_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "create_relations"


def test_tool_call_extractor_invalid_returns_empty():
    text = "No JSON here"
    assert ToolCallExtractor.extract_tool_calls(text) == []
