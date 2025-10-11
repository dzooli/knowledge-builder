from typing import Any, Dict, List

import pytest

from importer.src.processing.agent_orchestrator import AgentOrchestrator  # type: ignore


class MockConnector:
    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    def invoke_tool_by_name(
        self, name: str, params: Dict[str, Any]
    ):  # pragma: no cover - simple mock
        self.calls.append({"name": name, "params": params})
        # Simulate success results for write tools, echo for others
        if name in {"create_entities", "add_observations", "create_relations"}:
            return {"status": "ok", "tool": name}
        return {"status": "noop", "tool": name}

    async def ensure_mcp_tools(self):  # minimal placeholder for run_agent if needed
        return []


@pytest.fixture(name="orch")
def fixture_orchestrator():
    # type: ignore for passing mock instead of concrete Neo4jMemoryConnector
    return AgentOrchestrator(neo4j_connector=MockConnector())  # type: ignore[arg-type]


def test_make_evidence_name(orch):
    name = orch.make_evidence_name("SRC123", "0001")
    assert name == "Evidence SRC123-0001"


def test_build_evidence_entity(orch):
    entity = orch.build_evidence_entity("SRC123", "0001", "http://x")
    assert entity["name"] == "Evidence SRC123-0001"
    assert entity["type"] == "Evidence"
    assert any(
        "srcId=SRC123" in obs for obs in entity["observations"]
    )  # list contains src observation


def test_collect_capitalized_entities(orch):
    text = "OpenAI Created ChatGPT In San Francisco With Microsoft Investment"
    entities = orch.collect_capitalized_entities(
        text, max_scan_words=12, max_entities=4
    )
    assert entities  # not empty
    # Expect at least first capitalized token captured
    assert any("OpenAI" in e for e in entities)


def test_invoke_tool_success(orch):
    result, wrote = orch.invoke_tool_safe(
        "create_entities",
        {"entities": [{"name": "X", "type": "Concept", "observations": []}]},
    )
    assert wrote is True
    assert result["status"] == "ok"
    assert orch.neo4j_connector.calls[-1]["name"] == "create_entities"


def test_invoke_tool_non_write(orch):
    result, wrote = orch.invoke_tool_safe("read_graph", {"query": "abc"})
    assert wrote is False
    assert result["status"] == "noop"


def test_add_context_observation(orch):
    # Exercise helper indirectly by adding an observation to a fake entity
    orch.add_context_observation(
        "EntityA", "Some long text for observation", prefix="Context"
    )
    # Last call should be add_observations
    assert orch.neo4j_connector.calls[-1]["name"] == "add_observations"
