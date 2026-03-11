Data layout
-----------

Place per-company historical financial files (CSV or JSON) in this folder or a separate `data/` directory you point the evaluator at.

Expected filenames
- `{TICKER}.csv` or `{TICKER}.json` (e.g., `AAPL.csv`)

CSV schema (example columns)
- `period` (YYYY-MM-DD or YYYY)
- `revenue`
- `net_income`
- `total_debt`
- `shareholders_equity`
- `eps`
- `price`

The starter evaluator will attempt to read `{ticker}.csv` or `{ticker}.json` from the configured data directory.
