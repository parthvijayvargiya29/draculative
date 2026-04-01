"""Quantum feature transformer prototype.

This module provides a small Pennylane-based feature map prototype. If
`pennylane` is not available in the environment, it falls back to a
deterministic classical transform so the pipeline remains runnable.

Usage:
  from companies.src.quantum_features import quantum_transform_dataframe
  df_q = quantum_transform_dataframe(df, input_cols=['revenue','price','eps'])
"""
from typing import List
import numpy as np
import pandas as pd

try:
    import pennylane as qml
    _HAS_PENNYLANE = True
except Exception:
    _HAS_PENNYLANE = False


def _classical_fallback_transform(x: np.ndarray, n_out: int = 4) -> np.ndarray:
    """Simple deterministic transform used when PennyLane isn't installed.

    Produces sin/cos projections of normalized inputs.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.zeros(n_out)
    norm = (x - np.mean(x)) / (np.std(x) + 1e-8)
    out = []
    for i in range(n_out):
        out.append(np.tanh(np.sum(norm * np.sin((i + 1) * (np.arange(norm.size) + 1)))))
    return np.array(out)


if _HAS_PENNYLANE:
    # Small example quantum circuit using angle embedding and a single variational layer.
    def _build_qnode(n_qubits: int = 4):
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="autograd")
        def circuit(x, weights):
            # Angle embedding (repeat if input shorter than qubits)
            for i in range(n_qubits):
                qml.RY(x[i % len(x)], wires=i)
            # Simple variational rotations
            for i in range(n_qubits):
                qml.Rot(weights[i, 0], weights[i, 1], weights[i, 2], wires=i)
            # Entangle
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        return circuit


    def quantum_transform_row(x: List[float], n_qubits: int = 4, weights: np.ndarray = None) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        circuit = _build_qnode(n_qubits)
        if weights is None:
            # Small random seed for deterministic output during prototype
            rng = np.random.RandomState(42)
            weights = rng.normal(0, 0.1, size=(n_qubits, 3))
        return np.asarray(circuit(x, weights))


else:
    def quantum_transform_row(x: List[float], n_qubits: int = 4, weights: np.ndarray = None) -> np.ndarray:
        # Fallback: classical deterministic transform
        return _classical_fallback_transform(np.asarray(x, dtype=float), n_out=n_qubits)


def quantum_transform_dataframe(df: pd.DataFrame, input_cols: List[str], prefix: str = "q", n_qubits: int = 4) -> pd.DataFrame:
    """Apply quantum transform row-wise to the DataFrame and return appended columns.

    - `input_cols` selects which columns to encode into the circuit.
    - Returns a new DataFrame with additional columns `q0..q{n_qubits-1}`.
    """
    rows = []
    for _, r in df.iterrows():
        x = [float(r[c]) if pd.notna(r.get(c, np.nan)) else 0.0 for c in input_cols]
        q = quantum_transform_row(x, n_qubits=n_qubits)
        rows.append(q)
    qmat = np.vstack(rows)
    qcols = {f"{prefix}{i}": qmat[:, i] for i in range(qmat.shape[1])}
    qdf = pd.DataFrame(qcols, index=df.index)
    return pd.concat([df.reset_index(drop=True), qdf.reset_index(drop=True)], axis=1)
