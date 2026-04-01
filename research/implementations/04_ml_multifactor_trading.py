"""
ML Multi-Factor Quantitative Trading
=====================================
Paper: "Machine Learning Enhanced Multi-Factor Quantitative Trading:
        A Cross-Sectional Portfolio Optimization Approach with Bias Correction"
       Yimin Du (arXiv:2507.07107v1)

Core idea:
  Systematic alpha factor engineering + bias correction + cross-sectional
  portfolio optimization. Key innovations:
    - GBM data augmentation for limited training data
    - Cross-sectional neutralization (sector/market/style)
    - Multi-objective portfolio optimization (market-neutral, leverage-constrained)
    - Exponentially weighted factor covariance (EWMA risk model)

Key components implemented:
  1. Alpha factor computation (momentum, mean-reversion, volatility factors)
  2. Cross-sectional bias correction (z-score neutralization)
  3. GBM data augmentation
  4. Multi-factor risk model (Barra-style)
  5. Cross-sectional portfolio optimization (mean-variance, market neutral)
  6. EWMA factor covariance matrix
"""

import numpy as np
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# 1. Alpha Factor Computation
# ---------------------------------------------------------------------------

def factor_momentum(prices: np.ndarray, lookback: int = 21) -> np.ndarray:
    """
    Momentum factor: return over [lookback] days, skip last day.
    Cross-sectionally z-scored.

    Args:
        prices: (T, N) array of prices for N assets
        lookback: Lookback window in days (default 21 = 1 month)

    Returns:
        (T, N) factor values (NaN for first lookback days)
    """
    T, N = prices.shape
    factor = np.full((T, N), np.nan)
    for t in range(lookback, T):
        log_ret = np.log(prices[t - 1] / prices[t - lookback])  # skip last day
        factor[t] = _zscore_xs(log_ret)
    return factor


def factor_mean_reversion(prices: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Short-term mean reversion factor: negative of 5-day return.
    Assets that dropped recently are expected to rebound.

    Returns:
        (T, N) factor values (cross-sectionally z-scored, negated for reversal)
    """
    T, N = prices.shape
    factor = np.full((T, N), np.nan)
    for t in range(window, T):
        log_ret = np.log(prices[t] / prices[t - window])
        factor[t] = _zscore_xs(-log_ret)  # negated: lower return = higher score
    return factor


def factor_volatility(returns: np.ndarray, window: int = 21) -> np.ndarray:
    """
    Volatility factor: rolling realized volatility, negated (low vol preferred).

    Returns:
        (T, N) factor values
    """
    T, N = returns.shape
    factor = np.full((T, N), np.nan)
    for t in range(window, T):
        vol = returns[t - window: t].std(axis=0)
        factor[t] = _zscore_xs(-vol)  # negate: lower vol => higher score
    return factor


def factor_ewma_vol(returns: np.ndarray, halflife: int = 10) -> np.ndarray:
    """
    EWMA volatility factor (paper eq. 12).
    EWMAt = alpha * Xt + (1 - alpha) * EWMAt-1

    Args:
        halflife: Half-life in days for exponential weighting
    """
    T, N = returns.shape
    alpha = 1.0 - np.exp(-np.log(2) / halflife)
    factor = np.full((T, N), np.nan)
    ewma_var = returns[0] ** 2

    for t in range(1, T):
        ewma_var = alpha * returns[t] ** 2 + (1 - alpha) * ewma_var
        if t >= halflife:
            factor[t] = _zscore_xs(-np.sqrt(ewma_var))

    return factor


def _zscore_xs(values: np.ndarray, clip: float = 3.0) -> np.ndarray:
    """
    Cross-sectional z-score with outlier clipping.
    Removes market and style biases from a factor slice.
    """
    mean = np.nanmean(values)
    std = np.nanstd(values)
    if std < 1e-10:
        return np.zeros_like(values)
    z = (values - mean) / std
    return np.clip(z, -clip, clip)


# ---------------------------------------------------------------------------
# 2. Bias Correction — Cross-Sectional Neutralization
# ---------------------------------------------------------------------------

def neutralize_factor(
    factor: np.ndarray,
    exposures: np.ndarray,
) -> np.ndarray:
    """
    Remove systematic exposure from a factor via cross-sectional regression.
    This is the bias correction technique from the paper.

    For each time step t:
        Residual_t = Factor_t - X_t * (X_t'X_t)^{-1} X_t' * Factor_t

    where X_t are the neutralization exposures (e.g. sector dummies, beta, size).

    Args:
        factor:    (T, N) raw factor values
        exposures: (T, N, K) exposure matrix (K styles/sectors)

    Returns:
        (T, N) neutralized factor
    """
    T, N = factor.shape
    neutralized = np.full((T, N), np.nan)

    for t in range(T):
        f_t = factor[t]
        X_t = exposures[t]  # (N, K)
        mask = ~np.isnan(f_t) & ~np.any(np.isnan(X_t), axis=1)

        if mask.sum() < 5:
            neutralized[t] = f_t
            continue

        f_m = f_t[mask]
        X_m = X_t[mask]

        # OLS projection: fit = X (X'X)^{-1} X' f
        try:
            coef, _, _, _ = np.linalg.lstsq(X_m, f_m, rcond=None)
            fit = X_m @ coef
            residual = f_m - fit
            neutralized[t, mask] = _zscore_xs(residual)
        except np.linalg.LinAlgError:
            neutralized[t] = f_t

    return neutralized


# ---------------------------------------------------------------------------
# 3. GBM Data Augmentation (paper Section III.E)
# ---------------------------------------------------------------------------

def estimate_gbm_params(log_returns: np.ndarray) -> Tuple[float, float]:
    """
    Estimate GBM parameters via MLE (paper eq. 21-22).

    Args:
        log_returns: 1D array of log returns

    Returns:
        (mu_hat, sigma_hat) — drift and volatility estimates
    """
    n = len(log_returns)
    dt = 1.0  # daily

    sigma_hat = float(np.std(log_returns, ddof=1) / np.sqrt(dt))
    mu_hat = float(np.mean(log_returns) / dt + 0.5 * sigma_hat ** 2)

    return mu_hat, sigma_hat


def simulate_gbm(
    S0: float,
    mu: float,
    sigma: float,
    T: int,
    n_paths: int = 10,
    seed: int = 0,
) -> np.ndarray:
    """
    Simulate GBM price paths using Euler-Maruyama scheme (paper eq. 20).

    S_{t+dt} = S_t * exp((mu - sigma^2/2) * dt + sigma * sqrt(dt) * Z_t)

    Args:
        S0:      Initial price
        mu:      Drift (annualised if dt=1/252)
        sigma:   Volatility
        T:       Number of time steps
        n_paths: Number of synthetic paths

    Returns:
        (n_paths, T+1) array of simulated prices
    """
    rng = np.random.default_rng(seed)
    dt = 1.0  # daily step
    paths = np.zeros((n_paths, T + 1))
    paths[:, 0] = S0

    for t in range(T):
        Z = rng.standard_normal(n_paths)
        paths[:, t + 1] = paths[:, t] * np.exp(
            (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
        )

    return paths


def augment_returns(
    returns: np.ndarray,
    n_synthetic: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """
    Augment a return series with GBM-generated synthetic paths.

    Args:
        returns:     1D array of observed log returns
        n_synthetic: Number of synthetic paths to generate

    Returns:
        Augmented (1 + n_synthetic, T) array — row 0 is original
    """
    mu, sigma = estimate_gbm_params(returns)
    paths = simulate_gbm(
        S0=100.0,
        mu=mu,
        sigma=sigma,
        T=len(returns),
        n_paths=n_synthetic,
        seed=seed,
    )
    synth_returns = np.diff(np.log(paths), axis=1)  # (n_synthetic, T)
    return np.vstack([returns[np.newaxis, :], synth_returns])


# ---------------------------------------------------------------------------
# 4. EWMA Factor Covariance (paper eq. 29)
# ---------------------------------------------------------------------------

def ewma_covariance(
    factor_returns: np.ndarray,
    decay: float = 0.94,
) -> np.ndarray:
    """
    Compute EWMA factor covariance matrix (paper eq. 29):
        Omega_t = (1 - lambda) * sum_{s=0}^{inf} lambda^s * f_{t-s} f_{t-s}'

    Args:
        factor_returns: (T, K) factor return matrix
        decay:          Decay parameter lambda (0.94 = RiskMetrics standard)

    Returns:
        (K, K) EWMA covariance matrix
    """
    T, K = factor_returns.shape
    cov = np.zeros((K, K))

    for t in range(T):
        f = factor_returns[t].reshape(-1, 1)
        cov = decay * cov + (1 - decay) * (f @ f.T)

    return cov


# ---------------------------------------------------------------------------
# 5. Multi-Factor Risk Model (Barra-style)
# ---------------------------------------------------------------------------

def factor_risk_model(
    returns: np.ndarray,
    factor_loadings: np.ndarray,
    decay: float = 0.94,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decompose return covariance into factor + idiosyncratic components.
    (Paper eq. 28): Sigma = B * Omega * B' + Delta

    Args:
        returns:         (T, N) asset returns
        factor_loadings: (N, K) factor loading matrix B
        decay:           EWMA decay for factor covariance

    Returns:
        Sigma: (N, N) total covariance matrix
        Omega: (K, K) factor covariance
        Delta: (N,)   idiosyncratic variances
    """
    T, N = returns.shape
    N2, K = factor_loadings.shape
    assert N == N2

    B = factor_loadings

    # Factor returns via cross-sectional regression: F_t = (B'B)^{-1} B' r_t
    F = np.zeros((T, K))
    for t in range(T):
        r_t = returns[t]
        try:
            coef, _, _, _ = np.linalg.lstsq(B, r_t, rcond=None)
            F[t] = coef
        except np.linalg.LinAlgError:
            pass

    # EWMA factor covariance
    Omega = ewma_covariance(F, decay)

    # Idiosyncratic returns and variances
    idio = returns - (B @ F.T).T  # (T, N)
    Delta = np.var(idio, axis=0)

    # Total covariance
    Sigma = B @ Omega @ B.T + np.diag(Delta)

    return Sigma, Omega, Delta


# ---------------------------------------------------------------------------
# 6. Cross-Sectional Portfolio Optimization (paper eq. 23-27)
# ---------------------------------------------------------------------------

@dataclass
class PortfolioConstraints:
    """Constraints for cross-sectional portfolio optimization."""
    max_weight: float = 0.05      # Max absolute weight per asset (wmax)
    leverage:   float = 1.0       # Max sum of |weights| (L)
    risk_aversion: float = 1.0    # Lambda (risk penalty)
    tcost_aversion: float = 0.1   # Gamma (transaction cost penalty)
    market_neutral: bool = True   # Enforce sum(w) = 0


def optimize_portfolio(
    alpha: np.ndarray,
    Sigma: np.ndarray,
    prev_weights: Optional[np.ndarray] = None,
    tcosts: Optional[np.ndarray] = None,
    constraints: Optional[PortfolioConstraints] = None,
) -> np.ndarray:
    """
    Cross-sectional mean-variance portfolio optimization (paper eq. 23-27).

    Objective:
        max_w  w' * alpha - (lambda/2) * w' * Sigma * w - gamma * sum(c_i |w_i - w_{i,t-1}|)

    Subject to:
        sum(w_i) = 0       (market neutral)
        |w_i| <= wmax      (position limits)
        sum(|w_i|) <= L    (leverage)

    Solved via iterative projected gradient ascent (simple, interpretable).

    Args:
        alpha:        (N,) predicted returns / scores
        Sigma:        (N, N) covariance matrix
        prev_weights: (N,) previous period weights (for transaction cost)
        tcosts:       (N,) per-asset transaction costs c_i
        constraints:  PortfolioConstraints object

    Returns:
        (N,) optimal portfolio weights
    """
    N = len(alpha)
    if constraints is None:
        constraints = PortfolioConstraints()
    if prev_weights is None:
        prev_weights = np.zeros(N)
    if tcosts is None:
        tcosts = np.full(N, 0.001)

    # Initialise at equal-weight market neutral
    w = alpha / (np.abs(alpha).sum() + 1e-12)
    if constraints.market_neutral:
        w -= w.mean()

    lr = 0.1 / (N * constraints.risk_aversion + 1e-6)

    for step in range(300):
        # Gradient of objective
        grad_alpha = alpha
        grad_risk = constraints.risk_aversion * Sigma @ w
        grad_tcost = constraints.tcost_aversion * tcosts * np.sign(w - prev_weights)

        grad = grad_alpha - grad_risk - grad_tcost
        w = w + lr * grad

        # Project: market neutral
        if constraints.market_neutral:
            w -= w.mean()

        # Project: leverage (scale if too large)
        lev = np.abs(w).sum()
        if lev > constraints.leverage:
            w *= constraints.leverage / lev

        # Project: individual position limits
        w = np.clip(w, -constraints.max_weight, constraints.max_weight)

        # Re-enforce market neutral after clipping
        if constraints.market_neutral:
            w -= w.mean()

    return w


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  ML Multi-Factor Quantitative Trading")
    print("=" * 60)
    rng = np.random.default_rng(42)
    T, N = 252, 20

    # Synthetic prices (N assets)
    prices = np.exp(np.cumsum(rng.normal(0.0003, 0.015, (T, N)), axis=0)) * 100
    returns = np.diff(np.log(prices), axis=0)

    print(f"\n  [1] Alpha Factors (N={N} assets, T={T} days)")
    f_mom = factor_momentum(prices, lookback=21)
    f_rev = factor_mean_reversion(prices, window=5)
    f_vol = factor_volatility(returns, window=21)
    print(f"      Momentum factor (last row, mean): {np.nanmean(f_mom[-1]):.4f}")
    print(f"      Reversal factor (last row, mean): {np.nanmean(f_rev[-1]):.4f}")
    print(f"      Volatility factor (last row, std): {np.nanstd(f_vol[-1]):.4f}")

    print(f"\n  [2] GBM Augmentation (5 synthetic paths for asset 0)")
    mu_e, sig_e = estimate_gbm_params(returns[:, 0])
    print(f"      Estimated mu={mu_e*252:.2%}, sigma={sig_e*np.sqrt(252):.2%}")
    augmented = augment_returns(returns[:, 0], n_synthetic=5)
    print(f"      Augmented shape: {augmented.shape}")
    print(f"      Synthetic path 1 mean: {augmented[1].mean():.6f}")

    print(f"\n  [3] Factor Risk Model")
    K = 3
    B = rng.standard_normal((N, K))
    B /= np.linalg.norm(B, axis=0)
    Sigma, Omega, Delta = factor_risk_model(returns, B)
    print(f"      Factor cov matrix (K=3): diag = {np.diag(Omega).round(6)}")
    print(f"      Idio vol range: [{Delta.min():.4f}, {Delta.max():.4f}]")
    print(f"      Total cov trace: {np.trace(Sigma):.4f}")

    print(f"\n  [4] Portfolio Optimization (market neutral)")
    alpha_t = np.nanmean(
        np.stack([f_mom[-1], f_rev[-1], f_vol[-1]]), axis=0
    )
    weights = optimize_portfolio(alpha_t, Sigma)
    print(f"      Sum of weights (should ~0): {weights.sum():.6f}")
    print(f"      Leverage (|w| sum): {np.abs(weights).sum():.4f}")
    print(f"      Max abs weight: {np.abs(weights).max():.4f}")
    print(f"      Expected return: {float(weights @ alpha_t):.4f}")
    print(f"      Portfolio vol: {float(np.sqrt(weights @ Sigma @ weights)):.4f}")
