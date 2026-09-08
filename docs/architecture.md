# Architecture Deep Dive

```mermaid
flowchart TD
    %% DATA SOURCE
    subgraph SOURCE["DATA SOURCE"]
        D1["BatteryLife MATR Dataset"]
    end

    %% DATA & INGESTION
    subgraph DATA["DATA & INGESTION (Kafka)"]
        D2["canonical cycle / measurement data"]
        D3["deterministic progressive replay"]
        D4["Kafka telemetry + lifecycle events"]
        D2 --> D3
        D3 --> D4
    end

    D1 --> D2

    %% STREAMING & STATE
    subgraph STREAMING["STREAMING & STATE"]
        S1["Spark Structured Streaming"]
        S2["prefix-complete finalized state"]
        S3["persistent append-only Shared Feature Outlet"]
        D4 --> S1
        S1 --> S2
        S2 --> S3
    end

    %% ORCHESTRATION / MODELING (Airflow scope only from finalized state onward)
    subgraph ORCH["ORCHESTRATION / MODELING"]
        subgraph AIRFLOW["AIRFLOW-ORCHESTRATED"]
            A1["shared-state validation"]
            A2["generation receipt / cumulative generation ID selection"]
            A3["fixed validation/test benchmark"]
            A4["parallel RUL training/evaluation"]
            A5["parallel Survival training/evaluation"]
            A6["RUL candidate models"]
            A7["Survival candidate models"]
            A8["independent selection & publication"]
            A9["PostgreSQL loading"]

            A1 --> A2 --> A3
            A2 --> A4
            A2 --> A5
            A3 --> A4
            A3 --> A5
            A4 --> A6 --> A8
            A5 --> A7 --> A8
            A8 --> A9
        end
        S3 --> A1
    end

    %% SERVING
    subgraph SERVE["SERVING"]
        R1["newest finalized cumulative rows"]
        R2["Current RUL model"]
        R3["Current Survival model"]
        P1["PostgreSQL"]
        U1["Streamlit Dashboard"]
        S3 --> R1
        A2 --> R1
        A8 --> R2
        A8 --> R3
        R1 --> P1
        R2 --> P1
        R3 --> P1
        P1 --> U1
    end

    style SOURCE fill:#f8fafc,stroke:#374151,stroke-width:2px
```

## Detailed flow notes

- Shared Feature Outlet is the common upstream state for both model families.
- Generations consume cumulative finalized rows for training and selection.
- RUL and Survival diverge only after shared-state generation selection.
- Validation (and fixed benchmark) selects candidate family/configuration; test remains held out.
- Current RUL and Current Survival selections are independent.
- PostgreSQL is the serving boundary consumed by Streamlit.
