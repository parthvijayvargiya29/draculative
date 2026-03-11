Architecture — Company Fundamentals Evaluator
--------------------------------------------

High-level components

- Ingestion (streamed and batch): components that take raw data from sources (financial APIs, vendor CSVs, message queues) and normalize them to a canonical schema.
- Storage: a combination of a raw data lake (Parquet/S3 or local Parquet), a cleaned dataset (column-normalized Parquet / SQL tables) and optional feature store for computed metrics.
- Processing & Feature Engineering: compute time-series and cross-sectional metrics (revenue growth, margins, ratios, leverage), and store those as reusable features.
- Evaluation Engine: scoring logic that consumes features and configuration (weights, thresholds) to produce reproducible scores and explanations.
- Backtesting & Validation: run historical evaluations and simulate portfolios to validate scoring rules.
- APIs / UI: REST or simple CLI for requesting scores and visualizing results.
- Orchestration & Monitoring: workflow engine (Airflow/Prefect) and logging/metrics for data quality and pipeline health.

Data flow

1. Raw ingestion: pull data from provider APIs or uploaded CSVs. Store raw files with provenance and timestamps.
2. Normalization: map provider fields to canonical schema and perform basic cleaning.
3. Feature calculation: compute metrics for each period and company and store them in a feature store or table.
4. Evaluation: load features for target companies and apply scoring rules defined in configuration files.
5. Output: JSON or CSV reports, dashboards, and backtest results.

Tech stack recommendations (starter)

- Language: Python 3.10+
- Data libs: pandas, pyarrow (Parquet), numpy
- Storage: local Parquet for prototyping; PostgreSQL or DuckDB for queryable storage; S3 for scale
- Orchestration: Prefect for lightweight pipelines, Airflow if already in use
- Ingestion: requests, vendor SDKs; Kafka for high-throughput streams
- Feature store (optional): Feast or a simple table-based store in PostgreSQL/DuckDB
- Testing: pytest

Design notes

- Keep evaluation logic declarative: rules and weights in YAML so business users can iterate without code changes.
- Keep raw data immutable: never overwrite raw ingestion files; write normalized outputs to a separate location.
- Provide a reproducible CLI for one-off runs and a programmatic API for integration.

Next steps

- Create starter evaluator CLI
- Define canonical schema and example CSVs
- Add ingestion adapters for each data source
