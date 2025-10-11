import asyncio
from typing import Any, Dict, List

import pytest
from langchain_core.messages import AIMessage

from importer.src.processing.agent_orchestrator import AgentOrchestrator  # type: ignore


class MockConnector:
    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    def invoke_tool_by_name(
        self, name: str, params: Dict[str, Any]
    ):  # pragma: no cover
        # Record call
        self.calls.append({"name": name, "params": params})
        # Return a trivial structure (non-error) to simulate success
        return {"status": "ok", "tool": name}

    async def ensure_mcp_tools(self):  # pragma: no cover
        class _T:
            def __init__(self, name: str):
                self.name = name

        # Minimal tool list
        return [_T("create_entities"), _T("add_observations"), _T("create_relations")]


@pytest.fixture
def orch():
    return AgentOrchestrator(neo4j_connector=MockConnector())  # type: ignore[arg-type]


def test_run_with_timeout_success(orch):
    async def stub_run_agent(prompt: str):  # noqa: D401
        return {"messages": [AIMessage(content="Hello world", tool_calls=[])]}

    orch.run_agent = stub_run_agent  # type: ignore[assignment]
    messages = orch._run_with_timeout("PROMPT", timeout=1.0)  # noqa: SLF001
    assert messages and messages[0].content.startswith("Hello")


def test_run_with_timeout_timeout(orch):
    async def slow_run_agent(prompt: str):  # noqa: D401
        await asyncio.sleep(0.2)
        return {"messages": [AIMessage(content="Late", tool_calls=[])]}

    orch.run_agent = slow_run_agent  # type: ignore[assignment]
    messages = orch._run_with_timeout(
        "PROMPT", timeout=0.01
    )  # very small timeout  # noqa: SLF001
    assert messages == []


def test_process_chunk_forced_minimal_fallback(orch):
    # Return empty messages to force fallback path (no tool usage by AI)
    async def empty_run_agent(prompt: str):  # noqa: D401
        return {"messages": []}

    orch.run_agent = empty_run_agent  # type: ignore[assignment]

    source_id = "SRCX"
    chunk_id = "0001"
    text = "Acme Corp announced something in Paris."

    orch.process_chunk(source_id, chunk_id, "http://example", text)

    # Assert that a create_entities call for the evidence entity happened
    evidence_name = orch.make_evidence_name(source_id, chunk_id)
    create_calls = [
        c for c in orch.neo4j_connector.calls if c["name"] == "create_entities"
    ]
    assert create_calls, "Expected at least one create_entities call"
    assert any(
        any(evidence_name in str(v) for v in call["params"].values())
        for call in create_calls
    ), "Evidence entity not created in fallback path"

    # Ensure that either relations or observations were attempted
    relation_calls = [
        c for c in orch.neo4j_connector.calls if c["name"] == "create_relations"
    ]
    assert relation_calls, "Expected evidence relations in fallback path"
