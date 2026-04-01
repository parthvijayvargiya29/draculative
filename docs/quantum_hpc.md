# Quantum & HPC Integration Notes

This document captures a concise roadmap, install notes, and quick commands
to prototype quantum feature transforms and add HPC parallelism to the
`draculative` project.

1) Quantum prototype (Pennylane)

- Install (optional):
  - `python -m pip install pennylane`
- How to run the prototype we added:
  - The prototype transformer is in `companies/src/quantum_features.py`.
  - Basic smoke test (works with or without PennyLane):
    ```bash
    python - <<'PY'
    import pandas as pd
    from companies.src.quantum_features import quantum_transform_dataframe
    df = pd.DataFrame({'revenue':[100,200],'price':[10,12],'eps':[1.2,1.3]})
    print(quantum_transform_dataframe(df, ['revenue','price','eps']))
    PY
    ```

2) HPC: Dask / Ray batch runners

- A template Dask runner is in `predictor/scripts/dask_backtest.py`.
- Install Dask locally:
  - `python -m pip install 'dask[distributed]'`
- Run locally (example):
  - `python predictor/scripts/dask_backtest.py --tickers AAPL MSFT TSLA`
- For a cluster, start a scheduler/workers (or use Kubernetes) and pass
  `--scheduler-address tcp://SCHEDULER:8786`.

3) GPU acceleration

- RAPIDS (cuDF) can accelerate pandas-like operations on NVIDIA GPUs. It
  requires a compatible CUDA toolkit and drivers. See RAPIDS install guide:
  https://rapids.ai/start.html
- Alternatively, use JAX or PyTorch for numeric-heavy transforms and models.

4) Quantum optimization (portfolio/QAOA)

- Prototype idea: implement a small combinatorial QAOA optimizer for
  N<=20 assets and provide a classical fallback (simulated annealing or
  mixed-integer programming) if quantum resources are unavailable.

5) Next steps and priorities

- Wire `companies/src/quantum_features.py` into feature pipelines (done).
- Add GPU fast paths for heavy numeric transforms (optional, medium effort).
- Add distributed batch runner (Dask/Ray) for full backtests (template added).
- Prototype QAOA optimizer in a new module `companies/src/quantum_opt.py`.
