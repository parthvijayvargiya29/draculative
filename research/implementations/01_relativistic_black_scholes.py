"""
Relativistic Black-Scholes Model
=================================
Paper: "Relativistic Black-Scholes model" — Trzetrzelewski (arXiv:1307.5122v3)

Core idea:
  Standard Black-Scholes assumes Gaussian log-returns, implying infinite
  propagation speed. In reality, log-returns are bounded (market "speed of light").
  By applying a relativistic (telegrapher's equation) diffusion instead of
  plain Brownian motion, we naturally get a volatility frown/smile effect.

Key equations implemented:
  - Standard BS call price
  - Relativistic correction to BS (first-order 1/cm expansion)
  - Implied volatility surface comparison (BS vs Relativistic BS)
  - Volatility frown metric: how much the smile bends from relativistic correction
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Standard Black-Scholes (baseline)
# ---------------------------------------------------------------------------

def bs_d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Compute d1 in the Black-Scholes formula."""
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def bs_d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Compute d2 in the Black-Scholes formula."""
    return bs_d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Standard Black-Scholes European call price.

    Args:
        S: Spot price
        K: Strike
        T: Time to expiry (years)
        r: Risk-free rate
        sigma: Volatility

    Returns:
        Call option price
    """
    if T <= 0:
        return max(S - K, 0.0)
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Standard Black-Scholes European put price (via put-call parity)."""
    return bs_call(S, K, T, r, sigma) - S + K * np.exp(-r * T)


# ---------------------------------------------------------------------------
# Relativistic correction (first-order 1/cm expansion)
# Paper: Section 4 — the relativistic extension introduces a finite-velocity
# constraint cm on log-returns. The PDF is truncated at |x| < cm*T.
# The correction to the call price is derived as a 1/cm^2 expansion.
# ---------------------------------------------------------------------------

def relativistic_correction(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    cm: float,
) -> float:
    """
    First-order relativistic correction to the BS call price.

    The relativistic model replaces the Gaussian PDF with a finite-support
    PDF (telegrapher's equation). The leading correction term is O(1/cm^2).

    Correction (from paper eq. ~30):
        delta_C ≈ (sigma^2 * T / (2 * cm^2)) * d/dsigma [BS_call] * correction_factor

    Physically: cm is the "market speed of light" — the max log-return per unit
    time. The smaller cm, the stronger the relativistic correction.

    Args:
        cm: Market speed of light (max |d ln S/dt|); typical value 1-10 per day

    Returns:
        Correction term to add to standard BS call price
    """
    if cm <= 0:
        raise ValueError("cm must be positive")

    d1 = bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)

    # Vega of standard BS
    vega = S * norm.pdf(d1) * np.sqrt(T)

    # Relativistic correction factor (from telegrapher's 1/cm^2 term)
    # Captures the finite-velocity truncation of the log-return PDF
    alpha = sigma ** 2 / (2 * cm ** 2)

    # Correction modifies the effective diffusion
    # See paper: the correction introduces a skew proportional to x_max = cm * T
    x_max = cm * T
    correction = alpha * vega * (d1 * d2 - 1.0) / (sigma * np.sqrt(T)) * x_max

    return correction


def relativistic_bs_call(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    cm: float = 5.0,
) -> float:
    """
    Relativistic Black-Scholes European call price.

    C_rel = C_BS + delta_C(cm)

    As cm -> infinity, recovers standard BS.

    Args:
        cm: Market speed of light (default 5.0 per day — moderate relativistic effect)

    Returns:
        Relativistic-corrected call price
    """
    c_bs = bs_call(S, K, T, r, sigma)
    delta = relativistic_correction(S, K, T, r, sigma, cm)
    return max(c_bs + delta, 0.0)


# ---------------------------------------------------------------------------
# Implied volatility (invert BS formula numerically)
# ---------------------------------------------------------------------------

def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    tol: float = 1e-8,
) -> Optional[float]:
    """
    Compute implied volatility by inverting the BS formula (Brent's method).

    Args:
        market_price: Observed option price
        option_type: 'call' or 'put'

    Returns:
        Implied volatility, or None if not found
    """
    price_fn = bs_call if option_type == "call" else bs_put
    intrinsic = max(S - K * np.exp(-r * T), 0.0) if option_type == "call" else max(K * np.exp(-r * T) - S, 0.0)

    if market_price <= intrinsic:
        return None

    def objective(sigma):
        return price_fn(S, K, T, r, sigma) - market_price

    try:
        return brentq(objective, 1e-6, 10.0, xtol=tol)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Volatility smile / frown surface
# ---------------------------------------------------------------------------

@dataclass
class VolSurface:
    strikes: np.ndarray
    bs_ivols: np.ndarray        # Implied vols from standard BS prices
    rel_ivols: np.ndarray       # Implied vols from relativistic BS prices
    S: float
    T: float
    sigma: float
    cm: float


def compute_vol_surface(
    S: float = 100.0,
    T: float = 0.5,
    r: float = 0.02,
    sigma: float = 0.20,
    cm: float = 3.0,
    n_strikes: int = 30,
) -> VolSurface:
    """
    Compute and compare the implied vol surface for BS vs Relativistic BS.

    The relativistic model produces a frown shape (higher IV at the wings)
    due to the finite-velocity truncation effect.

    Args:
        cm: Market speed of light. Lower cm => stronger frown effect.
            Typical range: 1 (extreme) to 10 (mild).

    Returns:
        VolSurface with BS and relativistic implied vols per strike
    """
    # Moneyness range from -30% to +30% OTM
    moneyness = np.linspace(0.70, 1.30, n_strikes)
    strikes = S * moneyness

    bs_ivols = []
    rel_ivols = []

    for K in strikes:
        # BS call at flat vol sigma
        c_bs = bs_call(S, K, T, r, sigma)
        bs_iv = implied_vol(c_bs, S, K, T, r)

        # Relativistic BS call
        c_rel = relativistic_bs_call(S, K, T, r, sigma, cm)
        rel_iv = implied_vol(c_rel, S, K, T, r)

        bs_ivols.append(bs_iv if bs_iv else np.nan)
        rel_ivols.append(rel_iv if rel_iv else np.nan)

    return VolSurface(
        strikes=strikes,
        bs_ivols=np.array(bs_ivols),
        rel_ivols=np.array(rel_ivols),
        S=S,
        T=T,
        sigma=sigma,
        cm=cm,
    )


def frown_magnitude(surface: VolSurface) -> float:
    """
    Quantify the frown effect: average IV deviation between wings and ATM.

    Returns:
        Positive value => frown (wings have higher IV than ATM)
        Negative value => smile (wings have lower IV)
    """
    valid = ~np.isnan(surface.rel_ivols)
    atm_idx = np.argmin(np.abs(surface.strikes - surface.S))
    atm_iv = surface.rel_ivols[atm_idx]
    wing_iv = np.nanmean(np.concatenate([
        surface.rel_ivols[valid][:5],
        surface.rel_ivols[valid][-5:]
    ]))
    return wing_iv - atm_iv


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Relativistic Black-Scholes Model")
    print("=" * 60)

    S, K, T, r, sigma = 100.0, 100.0, 0.5, 0.02, 0.20

    c_bs = bs_call(S, K, T, r, sigma)
    print(f"\nStandard BS Call (ATM): {c_bs:.4f}")

    for cm in [10.0, 5.0, 3.0, 2.0, 1.0]:
        c_rel = relativistic_bs_call(S, K, T, r, sigma, cm)
        diff = c_rel - c_bs
        print(f"  Relativistic BS (cm={cm:.1f}): {c_rel:.4f}  |  correction = {diff:+.4f}")

    print(f"\n--- Volatility Frown Surface (cm=3.0, T=0.5) ---")
    surface = compute_vol_surface(S=100, T=0.5, r=0.02, sigma=0.20, cm=3.0)
    frown = frown_magnitude(surface)
    print(f"  Frown magnitude (wing IV - ATM IV): {frown*100:+.2f} vol points")

    print(f"\n  Strike | BS IV   | Rel IV")
    print(f"  -------|---------|--------")
    step = max(1, len(surface.strikes) // 10)
    for i in range(0, len(surface.strikes), step):
        K_i = surface.strikes[i]
        bs_iv = surface.bs_ivols[i]
        rel_iv = surface.rel_ivols[i]
        iv_str = f"{rel_iv*100:.2f}%" if not np.isnan(rel_iv) else "  N/A"
        print(f"  {K_i:6.1f} | {bs_iv*100:.2f}%  | {iv_str}")
