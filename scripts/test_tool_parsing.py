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

samples = []
# JSON array wrapped in fences
samples.append("""```json
[{"name":"create_entities","parameters":{"entities":[{"name":"X","type":"Thing","observations":[]}]}} , {"name":"add_observations","parameters":{"observations":[{"entityName":"X","observations":["hello"]}]}}]
```""")
# Single object
samples.append('{"name":"create_relations","parameters":{"source":"A","predicate":"offers","target":"B"}}')
# Embedded object in text
samples.append('Here is a tool call: {"name":"add_observations", "parameters": {"entityName": "Doc 1", "text": "[srcId=1 chunk=c1 url=http://x] evidence" } } And some trailing text.')

for i, s in enumerate(samples, 1):
    if i == 1:
        print("RAW#1:")
        print(s)
        print("CLEANED#1:")
        print(m.TextUtils.strip_code_fences(s))
    calls = m.ToolCallExtractor.extract_tool_calls(s)
    print(f"SAMPLE#{i}")
    for c in calls:
        print(c)
    print("-")
