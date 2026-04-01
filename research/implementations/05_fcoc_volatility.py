"""
FCOC: Fractal-Chaotic Oscillation Co-driven Framework for Volatility Forecasting
==================================================================================
Paper: "FCOC: A Fractal-Chaotic Co-driven Framework for Financial Volatility
        Forecasting" — Zeng et al. (arXiv:2511.10365v2)

Core idea:
  Two bottlenecks in standard DL volatility models:
    1. Feature fidelity: raw returns don't capture multifractal complexity
    2. Model responsiveness: static activation functions can't adapt dynamically

  Solutions:
    1. Fractal Feature Corrector (FFC): Overlapping Sliding Window Multifractal
       Asymmetric Detrended Cross-Correlation Analysis (OSW-MF-ADCCA)
    2. Chaotic Oscillation Component (COC): replace ReLU/tanh with dynamic
       chaotic activation functions (T9, T10 configurations)

Key components implemented:
  1. DFA (Detrended Fluctuation Analysis) — foundation
  2. MF-DFA (Multifractal DFA) — Hurst exponent spectrum
  3. MF-ADCCA with Overlapping Sliding Windows (FFC core)
  4. Multifractal spectrum (alpha, f(alpha))
  5. Chaotic activation functions (logistic map, tent map, T9, T10)
  6. Simple volatility forecaster using fractal features (COC demo)
"""

import numpy as np
from typing import Tuple, Optional, List


# ---------------------------------------------------------------------------
# 1. DFA — Detrended Fluctuation Analysis
# ---------------------------------------------------------------------------

def dfa(
    series: np.ndarray,
    scales: Optional[np.ndarray] = None,
    order: int = 1,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Detrended Fluctuation Analysis (DFA).
    Estimates the Hurst exponent H via power-law scaling of fluctuations.

    Algorithm:
      1. Integrate the series: y[t] = sum_{i=1}^{t} (x[i] - mean(x))
      2. For each scale s, divide y into non-overlapping windows
      3. Fit polynomial trend in each window, compute RMS residual F(s)
      4. H = slope of log F(s) vs log(s)

    Args:
        series: 1D time series (returns or prices)
        scales: Array of window sizes to use
        order:  Polynomial order for detrending (1 = linear)

    Returns:
        scales:  Window sizes
        F_s:     Fluctuation function values
        H:       Hurst exponent (0.5 = random walk, >0.5 = persistent)
    """
    x = series - np.mean(series)
    y = np.cumsum(x)  # profile

    if scales is None:
        n = len(y)
        scales = np.unique(np.logspace(
            np.log10(max(order + 2, 10)),
            np.log10(n // 4),
            20
        ).astype(int))

    F_s = []
    for s in scales:
        n_segments = len(y) // s
        if n_segments < 2:
            F_s.append(np.nan)
            continue

        segments = y[:n_segments * s].reshape(n_segments, s)
        t = np.arange(s)
        residuals = []

        for seg in segments:
            coef = np.polyfit(t, seg, order)
            trend = np.polyval(coef, t)
            residuals.append(np.mean((seg - trend) ** 2))

        F_s.append(np.sqrt(np.mean(residuals)))

    F_s = np.array(F_s)
    valid = ~np.isnan(F_s) & (F_s > 0)

    if valid.sum() >= 2:
        H = float(np.polyfit(np.log(scales[valid]), np.log(F_s[valid]), 1)[0])
    else:
        H = 0.5

    return np.array(scales), F_s, H


# ---------------------------------------------------------------------------
# 2. MF-DFA — Multifractal Detrended Fluctuation Analysis
# ---------------------------------------------------------------------------

def mfdfa(
    series: np.ndarray,
    q_values: Optional[np.ndarray] = None,
    scales: Optional[np.ndarray] = None,
    order: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Multifractal DFA — generalised fluctuation function for different q moments.

    F_q(s) = [1/N_s * sum_v |F_v(s)|^q]^{1/q}
    Scaling: F_q(s) ~ s^{h(q)}

    The generalised Hurst exponents h(q) characterise the multifractal spectrum.
    For a monofractal, h(q) = constant = H.

    Args:
        series:   1D time series
        q_values: Array of moment orders (default: -5 to 5)
        scales:   Window sizes

    Returns:
        q_values:  Moment orders
        h_q:       Generalised Hurst exponents
        F_q_all:   (len(q), len(scales)) fluctuation matrix
    """
    x = series - np.mean(series)
    y = np.cumsum(x)
    n = len(y)

    if q_values is None:
        q_values = np.array([-5, -4, -3, -2, -1, 0.1, 1, 2, 3, 4, 5], dtype=float)

    if scales is None:
        scales = np.unique(np.logspace(
            np.log10(max(order + 2, 10)),
            np.log10(n // 4),
            15
        ).astype(int))

    F_q_all = np.zeros((len(q_values), len(scales)))

    for si, s in enumerate(scales):
        n_segs = n // s
        if n_segs < 2:
            F_q_all[:, si] = np.nan
            continue

        segments = y[:n_segs * s].reshape(n_segs, s)
        t = np.arange(s)
        F2_segs = np.zeros(n_segs)

        for vi, seg in enumerate(segments):
            coef = np.polyfit(t, seg, order)
            trend = np.polyval(coef, t)
            F2_segs[vi] = np.mean((seg - trend) ** 2)

        for qi, q in enumerate(q_values):
            if abs(q) < 0.01:
                # q=0: use log average
                F_q_all[qi, si] = np.exp(0.5 * np.mean(np.log(F2_segs + 1e-20)))
            else:
                F_q_all[qi, si] = (np.mean(F2_segs ** (q / 2.0))) ** (1.0 / q)

    # Hurst exponents: h(q) = slope of log F_q(s) vs log(s)
    h_q = np.zeros(len(q_values))
    valid_s = ~np.any(np.isnan(F_q_all), axis=0)

    for qi in range(len(q_values)):
        fq = F_q_all[qi, valid_s]
        sv = scales[valid_s]
        if len(sv) >= 2 and np.all(fq > 0):
            h_q[qi] = float(np.polyfit(np.log(sv), np.log(fq), 1)[0])
        else:
            h_q[qi] = 0.5

    return q_values, h_q, F_q_all


# ---------------------------------------------------------------------------
# 3. Multifractal Spectrum (alpha, f(alpha))
# ---------------------------------------------------------------------------

def multifractal_spectrum(
    q_values: np.ndarray,
    h_q: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the multifractal spectrum (alpha, f(alpha)) via Legendre transform.

    From generalised Hurst exponents h(q):
        tau(q) = q * h(q) - 1
        alpha  = d tau / d q
        f(alpha) = q * alpha - tau(q)

    The width of the spectrum (Delta_alpha = alpha_max - alpha_min) measures
    multifractality strength. Wider => more complex/multifractal.

    Returns:
        alpha: Singularity strength (Holder exponents)
        f_alpha: Singularity spectrum
    """
    tau_q = q_values * h_q - 1.0

    # Numerical derivative
    alpha = np.gradient(tau_q, q_values)
    f_alpha = q_values * alpha - tau_q

    return alpha, f_alpha


def multifractal_width(alpha: np.ndarray, f_alpha: np.ndarray) -> float:
    """
    Spectral width Delta_alpha = alpha_max - alpha_min.
    Measures complexity: wider = more multifractal (more complex market dynamics).
    """
    valid = ~np.isnan(alpha) & ~np.isnan(f_alpha)
    if valid.sum() < 2:
        return 0.0
    return float(alpha[valid].max() - alpha[valid].min())


# ---------------------------------------------------------------------------
# 4. OSW-MF-ADCCA — Overlapping Sliding Window Multifractal
#    Asymmetric Detrended Cross-Correlation Analysis (FFC core)
# ---------------------------------------------------------------------------

def osw_mf_adcca(
    x: np.ndarray,
    y: np.ndarray,
    q_values: Optional[np.ndarray] = None,
    scales: Optional[np.ndarray] = None,
    order: int = 1,
    osw_step: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    OSW-MF-ADCCA: Overlapping Sliding Window Multifractal Asymmetric
    Detrended Cross-Correlation Analysis.

    Extension of MF-DFA to cross-correlations between two series,
    with asymmetry (separate treatment of positive/negative trends)
    and overlapping windows for stability (the FFC contribution from paper).

    Algorithm:
      1. Integrate both series
      2. For each scale s, use overlapping windows (step=osw_step < s)
      3. Detrend each window for both series
      4. Compute cross-correlation fluctuation F_xy^2(s,v)
      5. Separate into uptrend F+ and downtrend F- windows (asymmetry)
      6. Generalised cross-correlation exponent h_xy(q)

    Args:
        x, y:      1D time series (same length)
        osw_step:  Step for overlapping sliding windows (1 = maximum overlap)

    Returns:
        q_values: Moment orders
        h_xy_q:   Cross-correlation generalised Hurst exponents
    """
    assert len(x) == len(y)
    n = len(x)
    px = np.cumsum(x - x.mean())
    py = np.cumsum(y - y.mean())

    if q_values is None:
        q_values = np.array([-3, -2, -1, 0.1, 1, 2, 3], dtype=float)

    if scales is None:
        scales = np.unique(np.logspace(
            np.log10(max(order + 2, 10)),
            np.log10(n // 5),
            10,
        ).astype(int))

    h_xy_q = np.zeros(len(q_values))

    F_cross = np.zeros((len(q_values), len(scales)))

    for si, s in enumerate(scales):
        # Overlapping windows
        windows_x, windows_y = [], []
        for start in range(0, n - s + 1, osw_step):
            windows_x.append(px[start: start + s])
            windows_y.append(py[start: start + s])

        if len(windows_x) < 2:
            F_cross[:, si] = np.nan
            continue

        t = np.arange(s)
        F2_segs = []

        for wx, wy in zip(windows_x, windows_y):
            cx = np.polyfit(t, wx, order)
            cy = np.polyfit(t, wy, order)
            res_x = wx - np.polyval(cx, t)
            res_y = wy - np.polyval(cy, t)
            F2_segs.append(np.mean(res_x * res_y))

        F2_arr = np.array(F2_segs)
        # Asymmetric: separate positive and negative cross-correlations
        pos = F2_arr[F2_arr >= 0]
        neg = F2_arr[F2_arr < 0]

        for qi, q in enumerate(q_values):
            if len(pos) > 0 and len(neg) > 0:
                if abs(q) < 0.01:
                    fpos = np.exp(0.5 * np.mean(np.log(pos + 1e-20)))
                    fneg = np.exp(0.5 * np.mean(np.log(-neg + 1e-20)))
                else:
                    fpos = (np.mean(pos ** (q / 2.0))) ** (1.0 / q) if q > 0 else (np.mean(pos ** (q / 2.0))) ** (1.0 / q)
                    fneg = abs(np.mean((-neg) ** (abs(q) / 2.0))) ** (1.0 / abs(q))
                F_cross[qi, si] = 0.5 * (fpos + fneg)
            elif len(pos) > 0:
                arr = pos
                if abs(q) < 0.01:
                    F_cross[qi, si] = np.exp(0.5 * np.mean(np.log(arr + 1e-20)))
                else:
                    F_cross[qi, si] = float(np.mean(arr ** (q / 2.0 if q > 0 else 1)) ** (1.0 / max(q, 0.01)))
            else:
                F_cross[qi, si] = np.nan

    valid_s = ~np.any(np.isnan(F_cross), axis=0)
    for qi in range(len(q_values)):
        fq = F_cross[qi, valid_s]
        sv = scales[valid_s]
        if len(sv) >= 2 and np.all(fq > 0):
            h_xy_q[qi] = float(np.polyfit(np.log(sv), np.log(fq), 1)[0])
        else:
            h_xy_q[qi] = 0.5

    return q_values, h_xy_q


# ---------------------------------------------------------------------------
# 5. Chaotic Activation Functions — COC (Chaotic Oscillation Component)
# ---------------------------------------------------------------------------

def logistic_map(x: np.ndarray, r: float = 3.9) -> np.ndarray:
    """
    Logistic map activation: f(x) = r * sigmoid(x) * (1 - sigmoid(x))
    Exhibits chaotic dynamics for r > 3.57.

    Used as a dynamic activation in COC to replace static ReLU/tanh.
    """
    s = 1.0 / (1.0 + np.exp(-x))
    return r * s * (1.0 - s)


def tent_map(x: np.ndarray, mu: float = 1.9) -> np.ndarray:
    """
    Tent map activation: piecewise linear chaos.
    f(x) = mu * min(x_norm, 1 - x_norm) where x_norm = sigmoid(x)
    """
    s = 1.0 / (1.0 + np.exp(-x))
    return mu * np.minimum(s, 1.0 - s)


def coc_t9(x: np.ndarray, r: float = 3.9, alpha: float = 0.5) -> np.ndarray:
    """
    T9 chaotic activation (paper novel configuration):
    Combination of logistic map and tanh for richer dynamics.

    T9(x) = alpha * logistic_map(x) + (1 - alpha) * tanh(x)

    Args:
        r:     Logistic map parameter (chaoticity level)
        alpha: Blend weight between logistic and tanh
    """
    return alpha * logistic_map(x, r) + (1.0 - alpha) * np.tanh(x)


def coc_t10(x: np.ndarray, r: float = 3.8, beta: float = 0.3) -> np.ndarray:
    """
    T10 chaotic activation (paper novel configuration):
    Tent map + sigmoid combination for asymmetric chaos.

    T10(x) = beta * tent_map(x) + (1 - beta) * sigmoid(x)

    Args:
        r:    Tent map parameter
        beta: Blend weight
    """
    s = 1.0 / (1.0 + np.exp(-x))
    return beta * tent_map(x, r) + (1.0 - beta) * s


# ---------------------------------------------------------------------------
# 6. Fractal Feature Vector + Simple Volatility Forecaster
# ---------------------------------------------------------------------------

def extract_fractal_features(
    returns: np.ndarray,
    window: int = 100,
    step: int = 10,
) -> np.ndarray:
    """
    Extract rolling fractal features using FFC (OSW-MF-ADCCA) for a single asset.

    Features per window:
      - Hurst exponent H (DFA)
      - Multifractal width Delta_alpha
      - h(q=2) - h(q=-2): asymmetry measure

    Args:
        returns: 1D return series
        window:  Rolling window size
        step:    Step between windows

    Returns:
        (T_out, 3) feature matrix
    """
    n = len(returns)
    features = []
    times = []

    q_vals = np.array([-4, -2, -1, 0.1, 1, 2, 4], dtype=float)

    for end in range(window, n, step):
        seg = returns[end - window: end]
        _, F_s, H = dfa(seg)
        q, h_q, _ = mfdfa(seg, q_values=q_vals)
        alpha, f_alpha = multifractal_spectrum(q, h_q)
        width = multifractal_width(alpha, f_alpha)

        # Asymmetry: h(q>0) - h(q<0)
        h_pos = float(np.mean(h_q[q_vals > 0]))
        h_neg = float(np.mean(h_q[q_vals < 0]))
        asymmetry = h_pos - h_neg

        features.append([H, width, asymmetry])
        times.append(end)

    return np.array(features), np.array(times)


def forecast_volatility_with_fractals(
    returns: np.ndarray,
    horizon: int = 5,
    window: int = 100,
    step: int = 5,
) -> np.ndarray:
    """
    Simple volatility forecast using fractal features (COC demo).

    Method: Hurst exponent predicts next-period volatility scaling.
    H > 0.5 => trending (persistence) => higher future vol
    H < 0.5 => anti-persistent (mean-reverting) => lower future vol

    Returns:
        (T_out,) predicted volatility
    """
    features, times = extract_fractal_features(returns, window, step)
    H_vals = features[:, 0]

    # Realised vol over next [horizon] days
    vol_estimates = []
    for i, t in enumerate(times):
        if t + horizon <= len(returns):
            rv = float(np.std(returns[t: t + horizon]) * np.sqrt(252))
        else:
            rv = float(np.std(returns[max(0, t - horizon): t]) * np.sqrt(252))

        # Fractal-adjusted forecast: scale by Hurst deviation from random walk
        hurst_scale = 1.0 + 2.0 * (H_vals[i] - 0.5)  # 0.5 => scale=1.0, 1.0 => scale=2.0
        vol_estimates.append(rv * hurst_scale)

    return np.array(vol_estimates), times


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  FCOC: Fractal-Chaotic Volatility Framework")
    print("=" * 60)
    rng = np.random.default_rng(42)

    # Simulate returns with changing regime (persistent -> anti-persistent)
    n = 500
    returns = np.concatenate([
        np.cumsum(rng.normal(0, 0.01, 250)) * 0.01 + rng.normal(0, 0.015, 250),  # trending
        rng.normal(0, 0.02, 250),  # random walk
    ])

    print(f"\n  [1] DFA — Hurst Exponent")
    scales, F_s, H = dfa(returns)
    print(f"      Hurst exponent H = {H:.4f}  (0.5=random, >0.5=persistent)")

    print(f"\n  [2] MF-DFA — Multifractal Spectrum")
    q_vals = np.array([-4, -2, -1, 0.1, 1, 2, 4], dtype=float)
    q, h_q, _ = mfdfa(returns, q_values=q_vals)
    alpha, f_alpha = multifractal_spectrum(q, h_q)
    width = multifractal_width(alpha, f_alpha)
    print(f"      h(q) range: [{h_q.min():.3f}, {h_q.max():.3f}]")
    print(f"      Spectral width Δα = {width:.4f}  (wider => more multifractal)")

    print(f"\n  [3] OSW-MF-ADCCA Cross-Correlation")
    x = returns[:250]
    y = 0.7 * x + rng.normal(0, 0.005, 250)  # correlated asset
    q2, h_xy = osw_mf_adcca(x, y, q_values=q_vals[:5])
    print(f"      Cross-correlation h_xy(q=2): {h_xy[3]:.4f}")

    print(f"\n  [4] Chaotic Activation Functions")
    x_in = np.linspace(-2, 2, 5)
    print(f"      Input:   {x_in.round(2)}")
    print(f"      tanh:    {np.tanh(x_in).round(3)}")
    print(f"      COC-T9:  {coc_t9(x_in).round(3)}")
    print(f"      COC-T10: {coc_t10(x_in).round(3)}")

    print(f"\n  [5] Fractal Volatility Forecast")
    vol_forecast, times = forecast_volatility_with_fractals(returns, horizon=5, window=100)
    print(f"      Forecast points: {len(vol_forecast)}")
    print(f"      Mean forecast vol: {vol_forecast.mean():.2%} (annualised)")
    print(f"      Forecast range: [{vol_forecast.min():.2%}, {vol_forecast.max():.2%}]")
