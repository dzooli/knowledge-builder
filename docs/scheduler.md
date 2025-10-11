# Scheduler & Run Coordination

The scheduler ensures periodic, non-overlapping ingestion runs with graceful shutdown semantics.

## Responsibilities

* Periodic trigger (default every 5 minutes via `schedule` library)
* Overlap prevention via in-process run lock
* Early exit if stop event set (signal handler)
* Delegation to Document Processor for actual ETL work

## Sequence

```mermaid
sequenceDiagram
    autonumber
    participant Main as main.py
    participant S as Scheduler
    participant DP as DocumentProcessor
    participant SM as StateManager

    Main->>S: initialize(schedule_time)
    loop Every interval
        S->>S: is_running?
        alt Already Running
            S-->>S: Log skip
        else Not Running
            S->>S: acquire lock
            S->>DP: run_import()
            DP->>SM: load_state()
            DP->>DP: fetch_documents()
            DP->>DP: process_documents()
            DP->>SM: update_last_id/hash
            DP-->>S: run complete
            S->>S: release lock
        end
    end
```

## Locking Strategy

* Simple in-memory boolean / threading. Lock (sufficient for single-process container)
* Prevents scheduler drift causing overlapping long-running runs

## Shutdown

* Signal handler sets a stop event
* Scheduler loop checks stop flag before starting a new cycle
* In-flight document/chunk completes before process exits

## Future Enhancements

* External distributed lock (e.g., Redis) for horizontal scaling
* Metrics around run duration & skipped intervals
* Backoff / jitter to avoid thundering herd if scaled
