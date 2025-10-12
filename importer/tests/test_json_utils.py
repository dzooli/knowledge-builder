from importer.src.utils.json_utils import extract_first_json, safe_loads


def test_extract_first_json_simple_object():
    text = '{"a":1, "b":2}'
    parsed = extract_first_json(text)
    assert isinstance(parsed, dict)
    assert parsed["a"] == 1


def test_extract_first_json_array_in_fence():
    text = """Here is output:
```json
[{"name":"t","parameters":{}}]
```
Trailing text"""
    parsed = extract_first_json(text)
    assert isinstance(parsed, list)
    assert parsed and parsed[0]["name"] == "t"


def test_extract_first_json_with_marker_and_quotes():
    text = 'Stuff JSONSTART ```[{“name": "x", "parameters": {}}]``` end'
    parsed = extract_first_json(text)
    assert isinstance(parsed, list)
    assert parsed[0]["name"] == "x"


def test_safe_loads_fallback():
    # Validate that safe_loads works for bytes input
    data = b'{"k":"v"}'
    parsed = safe_loads(data)
    assert parsed["k"] == "v"
