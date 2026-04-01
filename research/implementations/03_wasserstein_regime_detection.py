"""
Sliced Wasserstein K-Means Market Regime Detection
===================================================
Paper: "Automated regime detection in multidimensional time series data
        using sliced Wasserstein k-means clustering"
       Luan & Hamp, Citigroup (arXiv:2310.01285v1)

Core idea:
  Treat rolling windows of returns as empirical distributions.
  Use Wasserstein distance (optimal transport) instead of Euclidean distance
  for k-means clustering — this is more sensitive to distributional shape shifts.
  Extend to multivariate data via sliced Wasserstein (random projections).

Key components implemented:
  1. Wasserstein distance (1D, p=2)
  2. Wasserstein barycentre (1D, p=2)
  3. Wasserstein k-means (Wk-means) — 1D
  4. Sliced Wasserstein distance (d>1)
  5. Sliced Wasserstein k-means (sWk-means) — multivariate
  6. Quality metrics (inertia, silhouette proxy)
  7. Regime labelling and transition detection
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# 1. Wasserstein distance (1D, p=2)
# ---------------------------------------------------------------------------

def wasserstein_1d(u: np.ndarray, v: np.ndarray, p: int = 2) -> float:
    """
    Compute the p-Wasserstein distance between two 1D empirical distributions.

    For 1D empirical distributions with equal mass atoms, the Wasserstein
    distance simplifies to (paper eq. 10):
        Wp^p(mu, nu) = (1/N) * sum_i |mu*_i - nu*_i|^p
    where mu*_i, nu*_i are the sorted atoms.

    Args:
        u: 1D array of samples from distribution mu
        v: 1D array of samples from distribution nu
        p: Order of Wasserstein distance (1 or 2)

    Returns:
        Wasserstein distance W_p(u, v)
    """
    n = min(len(u), len(v))
    u_sorted = np.sort(u)[:n]
    v_sorted = np.sort(v)[:n]
    return float((np.mean(np.abs(u_sorted - v_sorted) ** p)) ** (1.0 / p))


def wasserstein_barycentre_1d(distributions: List[np.ndarray], p: int = 2) -> np.ndarray:
    """
    Compute the Wasserstein barycentre of a family of 1D empirical distributions.

    From paper eq. (11-13):
      - For p=1: barycentre_j = Median(mu*_1j, ..., mu*_Mj)
      - For p=2: barycentre_j = Mean(mu*_1j, ..., mu*_Mj)

    Args:
        distributions: List of 1D sample arrays
        p: Order of Wasserstein distance

    Returns:
        Barycentre as sorted 1D sample array
    """
    n = min(len(d) for d in distributions)
    sorted_atoms = np.array([np.sort(d)[:n] for d in distributions])  # (M, n)

    if p == 1:
        return np.median(sorted_atoms, axis=0)
    else:  # p == 2 (default)
        return np.mean(sorted_atoms, axis=0)


# ---------------------------------------------------------------------------
# 2. Wasserstein K-Means (1D)
# ---------------------------------------------------------------------------

def wkmeans_1d(
    windows: List[np.ndarray],
    k: int = 3,
    max_iter: int = 100,
    n_init: int = 5,
    seed: int = 42,
    p: int = 2,
) -> Tuple[np.ndarray, List[np.ndarray], float]:
    """
    Wasserstein k-means clustering of 1D empirical distributions.

    Algorithm (paper Section 2.3):
      1. Initialise k centroids by random selection from windows
      2. Assign each window to nearest centroid (Wasserstein distance)
      3. Update centroids as Wasserstein barycentres of assigned windows
      4. Repeat until convergence

    Args:
        windows: List of 1D sample arrays (each = one time window's returns)
        k:       Number of clusters (regimes)
        n_init:  Number of restarts (take best inertia)
        p:       Wasserstein order

    Returns:
        labels:    Cluster assignment per window
        centroids: Centroid distributions
        inertia:   Total within-cluster Wasserstein distance
    """
    rng = np.random.default_rng(seed)
    best_labels, best_centroids, best_inertia = None, None, np.inf

    for init in range(n_init):
        # Random initialisation
        idx = rng.choice(len(windows), size=k, replace=False)
        centroids = [windows[i].copy() for i in idx]

        labels = np.zeros(len(windows), dtype=int)
        for iteration in range(max_iter):
            # Assignment step
            new_labels = np.zeros(len(windows), dtype=int)
            for i, w in enumerate(windows):
                dists = [wasserstein_1d(w, c, p) for c in centroids]
                new_labels[i] = int(np.argmin(dists))

            # Check convergence
            if np.all(new_labels == labels):
                break
            labels = new_labels

            # Update step: barycentres
            for j in range(k):
                members = [windows[i] for i in range(len(windows)) if labels[i] == j]
                if members:
                    centroids[j] = wasserstein_barycentre_1d(members, p)

        # Compute inertia
        inertia = sum(
            wasserstein_1d(windows[i], centroids[labels[i]], p) ** p
            for i in range(len(windows))
        )

        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centroids = [c.copy() for c in centroids]

    return best_labels, best_centroids, best_inertia


# ---------------------------------------------------------------------------
# 3. Sliced Wasserstein Distance (d > 1)
# ---------------------------------------------------------------------------

def sliced_wasserstein(
    X: np.ndarray,
    Y: np.ndarray,
    n_projections: int = 50,
    p: int = 2,
    seed: int = 0,
) -> float:
    """
    Approximate the Wasserstein distance between two multivariate distributions
    using the sliced Wasserstein method (paper Section 2.2.4).

    Algorithm:
      1. Sample random unit vectors theta on the (d-1)-sphere
      2. Project both distributions onto each theta
      3. Compute 1D Wasserstein distance for each projection
      4. Average over all projections

    From paper eq. (16-17):
        SW_p^p(mu, nu) = integral_{S^{d-1}} W_p^p(mu'(theta), nu'(theta)) d_theta

    Args:
        X: (n, d) array — samples from distribution mu
        Y: (m, d) array — samples from distribution nu
        n_projections: Number of random projections (L in paper)
        p: Wasserstein order

    Returns:
        Sliced Wasserstein distance
    """
    assert X.ndim == 2 and Y.ndim == 2, "X and Y must be 2D arrays"
    d = X.shape[1]
    rng = np.random.default_rng(seed)

    # Sample random directions on unit sphere
    thetas = rng.standard_normal((n_projections, d))
    thetas /= np.linalg.norm(thetas, axis=1, keepdims=True)

    total = 0.0
    for theta in thetas:
        # Project onto 1D
        px = X @ theta
        py = Y @ theta
        total += wasserstein_1d(px, py, p) ** p

    return float((total / n_projections) ** (1.0 / p))


def sliced_wasserstein_barycentre(
    distributions: List[np.ndarray],
    n_projections: int = 50,
    p: int = 2,
    n_iter: int = 20,
    seed: int = 0,
) -> np.ndarray:
    """
    Compute sliced Wasserstein barycentre for multivariate distributions.

    Approximation: project to many 1D spaces, average barycentres per projection,
    then reconstruct using gradient-based fixed-point iteration.
    For simplicity, we use the coordinate-wise mean as an efficient approximation.

    Returns:
        (n, d) array — barycentre samples
    """
    n = min(len(d) for d in distributions)
    d = distributions[0].shape[1]
    rng = np.random.default_rng(seed)

    # Coordinate-wise Wasserstein barycentre (simple approximation)
    bary = np.zeros((n, d))
    for dim in range(d):
        cols = [np.sort(dist[:n, dim]) for dist in distributions]
        bary[:, dim] = np.mean(cols, axis=0)

    return bary


# ---------------------------------------------------------------------------
# 4. Sliced Wasserstein K-Means (sWk-means) — multivariate
# ---------------------------------------------------------------------------

def swkmeans(
    windows: List[np.ndarray],
    k: int = 3,
    max_iter: int = 50,
    n_init: int = 3,
    n_projections: int = 50,
    p: int = 2,
    seed: int = 42,
) -> Tuple[np.ndarray, List[np.ndarray], float]:
    """
    Sliced Wasserstein k-means clustering for multivariate distributions.
    Extends Wk-means to d > 1 using sliced Wasserstein distance.

    Args:
        windows: List of (n_obs, d) arrays — one per time window
        k:       Number of regimes
        n_projections: Number of random projections for SW distance

    Returns:
        labels:    Cluster assignment per window
        centroids: Centroid distributions (each a (n, d) array)
        inertia:   Total within-cluster SW distance
    """
    rng = np.random.default_rng(seed)
    best_labels, best_centroids, best_inertia = None, None, np.inf

    for init in range(n_init):
        idx = rng.choice(len(windows), size=k, replace=False)
        centroids = [windows[i].copy() for i in idx]
        labels = np.zeros(len(windows), dtype=int)

        for iteration in range(max_iter):
            new_labels = np.zeros(len(windows), dtype=int)
            for i, w in enumerate(windows):
                dists = [sliced_wasserstein(w, c, n_projections, p, seed + i) for c in centroids]
                new_labels[i] = int(np.argmin(dists))

            if np.all(new_labels == labels):
                break
            labels = new_labels

            for j in range(k):
                members = [windows[i] for i in range(len(windows)) if labels[i] == j]
                if members:
                    centroids[j] = sliced_wasserstein_barycentre(members, n_projections, p, seed=seed)

        inertia = sum(
            sliced_wasserstein(windows[i], centroids[labels[i]], n_projections, p) ** p
            for i in range(len(windows))
        )

        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centroids = [c.copy() for c in centroids]

    return best_labels, best_centroids, best_inertia


# ---------------------------------------------------------------------------
# 5. Utility: build rolling windows from a return series
# ---------------------------------------------------------------------------

def build_rolling_windows(
    returns: np.ndarray,
    window_size: int = 20,
    step: int = 5,
) -> List[np.ndarray]:
    """
    Build a list of rolling windows from a return series.

    Args:
        returns:     1D (univariate) or 2D (multivariate, shape n x d) array
        window_size: Observations per window
        step:        Step between windows

    Returns:
        List of (window_size,) or (window_size, d) arrays
    """
    n = len(returns)
    windows = []
    for start in range(0, n - window_size + 1, step):
        windows.append(returns[start: start + window_size])
    return windows


def regime_transitions(labels: np.ndarray) -> List[Tuple[int, int, int]]:
    """
    Identify regime change points.

    Returns:
        List of (window_index, from_regime, to_regime)
    """
    transitions = []
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            transitions.append((i, int(labels[i - 1]), int(labels[i])))
    return transitions


def regime_stats(returns: np.ndarray, labels: np.ndarray, k: int) -> dict:
    """
    Compute per-regime statistics (mean return, volatility, skew).

    Returns:
        Dict mapping regime_id -> {mean, vol, skew, count}
    """
    # labels index windows (len = n_windows); returns is the full series.
    # Map window index -> return index via step and window_size alignment.
    # Use the label repeated proportionally to the number of observations.
    stats = {}
    # Trim returns to match the number of labels (one label per window)
    r_flat = returns if returns.ndim == 1 else returns[:, 0]
    n_labels = len(labels)
    step = max(1, len(r_flat) // n_labels)
    # Build a per-observation label array by stretching window labels
    obs_labels = np.repeat(labels, step)
    # Align lengths: trim the longer one
    min_len = min(len(r_flat), len(obs_labels))
    r_flat = r_flat[:min_len]
    obs_labels = obs_labels[:min_len]

    for j in range(k):
        mask = obs_labels == j
        r = r_flat[mask]
        if len(r) > 1:
            stats[j] = {
                "mean":  float(np.mean(r)),
                "vol":   float(np.std(r)),
                "skew":  float(((r - r.mean()) ** 3).mean() / (r.std() ** 3 + 1e-12)),
                "count": int(mask.sum()),
            }
    return stats


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Sliced Wasserstein Regime Detection")
    print("=" * 60)
    rng = np.random.default_rng(0)

    # Simulate 3 regimes: bull (low vol positive), bear (high vol negative), crisis
    n = 400
    returns = np.concatenate([
        rng.normal(0.05 / 252, 0.010, 120),  # bull
        rng.normal(-0.10 / 252, 0.025, 100), # bear
        rng.normal(0.00,        0.050, 80),  # crisis
        rng.normal(0.03 / 252,  0.012, 100), # recovery
    ])

    windows = build_rolling_windows(returns, window_size=20, step=5)
    print(f"\n  Total windows: {len(windows)}")
    print(f"  Running Wasserstein k-means (k=3)...")

    labels, centroids, inertia = wkmeans_1d(windows, k=3, n_init=5, seed=42)
    stats = regime_stats(returns, labels, k=3)

    print(f"  Inertia: {inertia:.4f}")
    transitions = regime_transitions(labels)
    print(f"  Regime transitions: {len(transitions)}")

    print(f"\n  Regime stats:")
    for regime_id, s in stats.items():
        print(f"    Regime {regime_id}: mean={s['mean']*252:.2f}% ann | vol={s['vol']*np.sqrt(252):.1%} | count={s['count']}")

    # Multivariate test (2 assets)
    print(f"\n  Running sWk-means on 2-asset returns (k=3)...")
    ret2d = np.column_stack([
        returns,
        returns * 0.6 + rng.normal(0, 0.005, len(returns))
    ])
    windows_2d = build_rolling_windows(ret2d, window_size=20, step=5)
    labels_2d, _, inertia_2d = swkmeans(windows_2d, k=3, n_init=3, n_projections=30, seed=42)
    transitions_2d = regime_transitions(labels_2d)
    print(f"  2D Inertia: {inertia_2d:.4f}")
    print(f"  2D Regime transitions: {len(transitions_2d)}")
    print(f"  Labels (first 20): {labels_2d[:20].tolist()}")
