import sys
import types

# Mock external deps so importer can be imported in this environment
mods = {
    'langchain_ollama': types.ModuleType('langchain_ollama'),
    'langchain_mcp_adapters': types.ModuleType('langchain_mcp_adapters'),
    'langchain_mcp_adapters.client': types.ModuleType('langchain_mcp_adapters.client'),
    'langgraph': types.ModuleType('langgraph'),
    'langgraph.prebuilt': types.ModuleType('langgraph.prebuilt'),
    'langchain_core': types.ModuleType('langchain_core'),
    'langchain_core.messages': types.ModuleType('langchain_core.messages'),
    'langchain_core.tools': types.ModuleType('langchain_core.tools'),
}

# Minimal class stubs used by importer
class ChatOllama:
    def __init__(self, *args, **kwargs):
        pass

class MultiServerMCPClient:
    def __init__(self, *args, **kwargs):
        pass
    async def get_tools(self):
        return []

def create_react_agent(model, tools):
    class Dummy:
        async def ainvoke(self, state):
            return {"messages": []}
    return Dummy()

class AIMessage:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []

class ToolMessage:
    def __init__(self, name="", content=""):
        self.name = name
        self.content = content

class BaseTool:
    name = "dummy"

mods['langchain_ollama'].ChatOllama = ChatOllama
mods['langchain_mcp_adapters'].client = mods['langchain_mcp_adapters.client']
mods['langchain_mcp_adapters.client'].MultiServerMCPClient = MultiServerMCPClient
mods['langgraph'].prebuilt = mods['langgraph.prebuilt']
mods['langgraph.prebuilt'].create_react_agent = create_react_agent
mods['langchain_core'].messages = mods['langchain_core.messages']
mods['langchain_core.messages'].AIMessage = AIMessage
mods['langchain_core.messages'].ToolMessage = ToolMessage
mods['langchain_core'].tools = mods['langchain_core.tools']
mods['langchain_core.tools'].BaseTool = BaseTool

for k, v in mods.items():
    sys.modules[k] = v

sys.path.insert(0, r"D:\projects\ai\knowledge-builder\importer\src")
import importer as m

# Prepare synthetic tool calls covering all branches
calls = [
    {"name": "add_observations", "parameters": {"observations": [
        {"entityName": "Doc 1", "observations": ["text"]},
        {"entityName": {"name": "Doc 2"}, "observations": []},
        {"entityName": 123}  # ignored
    ]}},
    {"name": "create_entities", "parameters": {"entities": [
        {"name": "Company A", "type": "Organization"},
        {"name": "Company B"},
        "bad"  # ignored
    ]}},
    {"name": "create_relations", "parameters": {"relations": [
        {"source": "Company A", "relationType": "offers", "target": "Product X"},
        {"source": {"name": "Company B"}, "relationType": "partners_with", "target": {"name": "Company C"}},
        {"source": None, "relationType": "x", "target": "Y"}  # ignored in name extraction
    ]}},
    {"name": "unknown", "parameters": {}},
    "bad_call"
]

orchestrator = m.AgentOrchestrator(m.Neo4jMemoryConnector())
extracted = orchestrator.extract_entities_from_calls(calls)
print("EXTRACTED:", extracted)

# Expect deduped order-preserving list
expected_order = [
    "Doc 1", "Doc 2", "Company A", "Company B", "Company A", "Product X", "Company B", "Company C"
]
# After dedupe, expected unique order
expected_unique = []
seen = set()
for n in expected_order:
    if n not in seen:
        expected_unique.append(n)
        seen.add(n)

print("EXPECTED_UNIQUE:", expected_unique)
print("PASS:", extracted == expected_unique)

