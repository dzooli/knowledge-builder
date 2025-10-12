# Knowledge Builder Docs

This project is WIP, contributors are welcome.

## High-Level Flow

```mermaid
sequenceDiagram
    autonumber
    participant DP as DocumentProcessor
    participant AO as AgentOrchestrator
    participant LLM as ReAct Agent (Ollama)
    participant MCP as Neo4j Memory MCP
    participant N4J as Neo4j

    DP->>AO: process_chunk(text_chunk)
    AO->>AO: Stage 1: Primary prompt & timeout
    AO->>LLM: Run ReAct agent
    LLM-->>AO: Structured / freeform output
    AO->>AO: Parse tool calls
    AO->>AO: Stage 2: Execute AI tool calls
    loop For each parsed call
        AO->>MCP: invoke_tool(name, params)
        MCP->>N4J: Graph mutation / query
        N4J-->>MCP: Result
        MCP-->>AO: Tool result
    end
    AO->>AO: Sufficiency check
    alt AI output insufficient
        AO->>AO: Stage 3: Heuristic augmentation
        AO->>AO: Extract entities from text
        AO->>MCP: create_entities / add_observations
        MCP->>N4J: Writes
    end
    alt Still insufficient
        AO->>AO: Stage 4: Forced minimal write
        AO->>MCP: Ensure Evidence + GenericTopic
    end
    AO->>AO: Stage 5: Finalize & summarize
    AO-->>DP: Executed tool call list
```

## Key Guarantees

* Deterministic Evidence node per chunk
* Progressive enrichment ladder (AI → heuristic → forced)
* Bounded agent time (single timeout wrapper)
* Centralized logging & tool call execution

## More Details

See the full design in `agent_orchestrator.md` .
