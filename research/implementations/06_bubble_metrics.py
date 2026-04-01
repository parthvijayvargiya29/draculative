"""
Experimental Asset Market — Testosterone & Bubble Metrics
==========================================================
Paper: "The Bull of Wall Street: Experimental Analysis of Testosterone
        and Asset Trading" — Nadler, Jiao, Johnson, Alexander & Zak

Core idea:
  Testosterone causally increases overbidding in experimental asset markets
  (Smith, Suchanek & Williams 1988 framework). Testosterone-treated traders
  bid higher relative to fundamental value, creating larger and longer bubbles.

  Key metrics from paper:
    - Bubble Amplitude: average deviation of transaction price from fundamental value
    - Bubble Duration: length of the period prices exceed fundamental value
    - Turnover: total shares traded / total shares outstanding (trading activity)
    - Overbidding: bids above fundamental value
    - Price Deviation Index (RAD / RD)

Key components implemented:
  1. SSW experimental market simulator (fundamental value decay)
  2. Bubble detection and measurement (amplitude, duration, turnover)
  3. Overbidding model (testosterone effect as bid premium)
  4. RAD and RD price deviation metrics
  5. Market simulation: control vs testosterone group
  6. Statistical comparison of bubble metrics
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# 1. SSW Experimental Market Setup
# ---------------------------------------------------------------------------

@dataclass
class SSWMarket:
    """
    Smith-Suchanek-Williams (1988) experimental asset market.

    Structure:
      - T trading periods (typically 15)
      - Each period: asset pays a dividend d with probability p, else 0
      - Fundamental value at period t: FV_t = (T - t) * E[dividend]
        (expected remaining dividends)
      - Asset expires worthless at T

    Attributes:
        n_periods:       Total trading periods
        dividend_probs:  Probability of dividend each period [p_low, p_high]
        dividends:       Possible dividend amounts [d_low, d_high]
        n_shares:        Total shares outstanding
        n_traders:       Number of traders in the market
    """
    n_periods: int = 15
    dividend_probs: Tuple[float, float] = (0.5, 0.5)
    dividends: Tuple[float, float] = (0.0, 0.28)       # $0 or $0.28
    n_shares: int = 10
    n_traders: int = 6
    initial_cash: float = 5.0                           # per trader
    initial_shares: int = 2                             # per trader (if n_shares=n_traders*2)

    @property
    def expected_dividend(self) -> float:
        """Expected dividend per period."""
        return sum(p * d for p, d in zip(self.dividend_probs, self.dividends))

    def fundamental_value(self, period: int) -> float:
        """
        Fundamental value at start of period t (0-indexed).
        FV_t = (T - t) * E[div]

        At period 0 (start): FV = T * E[div]
        At period T-1 (last): FV = E[div]
        After last period: FV = 0
        """
        remaining = self.n_periods - period
        return max(0.0, remaining * self.expected_dividend)

    def fundamental_value_series(self) -> np.ndarray:
        """Return fundamental values for all periods."""
        return np.array([self.fundamental_value(t) for t in range(self.n_periods + 1)])


# ---------------------------------------------------------------------------
# 2. Overbidding Model — Testosterone Effect
# ---------------------------------------------------------------------------

def testosterone_bid_premium(
    testosterone_level: float,
    base_premium: float = 0.0,
    sensitivity: float = 0.15,
) -> float:
    """
    Model the testosterone-induced bid premium above fundamental value.

    From paper findings:
      - Testosterone-treated traders bid significantly higher than fundamental value
      - Placebo group bids closer to fundamental value (or slightly above)
      - Effect is continuous with testosterone level

    Args:
        testosterone_level: Standardised testosterone level (z-score relative to baseline)
        base_premium:       Base overbidding premium (0 = placebo group)
        sensitivity:        How much each unit of T increases premium

    Returns:
        Bid premium fraction above FV (e.g. 0.20 => 20% above FV)
    """
    return base_premium + sensitivity * max(testosterone_level, 0)


def generate_bids(
    fundamental_value: float,
    n_traders: int,
    testosterone_levels: np.ndarray,
    noise_std: float = 0.05,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate bids from traders with different testosterone levels.

    Each trader bids:
        bid_i = FV * (1 + premium_i + noise_i)

    where premium_i is driven by their testosterone level.

    Args:
        testosterone_levels: (n_traders,) standardised T levels
        noise_std:           Random variation in bids (idiosyncratic)

    Returns:
        (n_traders,) bid array
    """
    rng = np.random.default_rng(seed)
    bids = []
    for t_level in testosterone_levels:
        premium = testosterone_bid_premium(t_level)
        noise = rng.normal(0, noise_std)
        bid = fundamental_value * (1 + premium + noise)
        bids.append(max(bid, 0.0))
    return np.array(bids)


# ---------------------------------------------------------------------------
# 3. Market Simulation
# ---------------------------------------------------------------------------

@dataclass
class MarketResult:
    """Stores results from one market simulation run."""
    periods: np.ndarray
    fundamental_values: np.ndarray
    transaction_prices: np.ndarray
    bids: np.ndarray             # (T, n_traders)
    volume: np.ndarray           # shares traded per period
    dividends_paid: np.ndarray   # dividend realised each period
    treatment: str               # 'placebo' or 'testosterone'


def simulate_market(
    market: SSWMarket,
    testosterone_levels: np.ndarray,
    treatment: str = "placebo",
    price_adjustment_speed: float = 0.5,
    seed: int = 42,
) -> MarketResult:
    """
    Simulate one SSW experimental market over T periods.

    Price discovery mechanism:
      - Each period: traders submit bids
      - Transaction price = weighted average of top (willing) bids
      - Price adjusts toward bid midpoint from previous period

    Args:
        testosterone_levels: (n_traders,) standardised T levels
        treatment:           Label for the group
        price_adjustment_speed: How quickly price adjusts to bids (0-1)

    Returns:
        MarketResult with full history
    """
    rng = np.random.default_rng(seed)
    T = market.n_periods
    n = market.n_traders

    fv_series = market.fundamental_value_series()
    prices = np.zeros(T)
    bids_all = np.zeros((T, n))
    volume = np.zeros(T)
    dividends = np.zeros(T)

    # Start price near fundamental value with noise
    price = fv_series[0] * rng.uniform(0.9, 1.1)

    for t in range(T):
        fv = fv_series[t]

        # Generate bids
        bids_t = generate_bids(fv, n, testosterone_levels, noise_std=0.04, seed=seed + t)
        bids_all[t] = bids_t

        # Price discovery: transaction price = trimmed mean of top 50% bids
        top_bids = np.sort(bids_t)[-max(n // 2, 1):]
        target_price = np.mean(top_bids)

        # Price adjustment (partial adjustment model)
        price = price + price_adjustment_speed * (target_price - price)
        price = max(price, 0.0)
        prices[t] = price

        # Volume: higher price dispersion => more trading
        spread = np.std(bids_t)
        volume[t] = rng.poisson(max(1, int(spread * n * 2)))

        # Dividend realisation
        div = rng.choice(market.dividends, p=market.dividend_probs)
        dividends[t] = div

    return MarketResult(
        periods=np.arange(T),
        fundamental_values=fv_series[:T],
        transaction_prices=prices,
        bids=bids_all,
        volume=volume,
        dividends_paid=dividends,
        treatment=treatment,
    )


# ---------------------------------------------------------------------------
# 4. Bubble Metrics (paper Section III)
# ---------------------------------------------------------------------------

def bubble_amplitude(result: MarketResult) -> float:
    """
    Bubble amplitude: average (price - FV) across all periods.

    Paper metric: measures how far prices deviate above fundamental value on average.
    Positive => prices exceeded fundamental value (bubble).

    Returns:
        Mean price - FV (in dollars)
    """
    deviations = result.transaction_prices - result.fundamental_values
    return float(np.mean(deviations))


def bubble_duration(result: MarketResult) -> int:
    """
    Bubble duration: number of consecutive periods where price > FV.

    Paper metric: counts the maximum run of periods where prices stayed above FV.

    Returns:
        Max consecutive periods above fundamental value
    """
    above = result.transaction_prices > result.fundamental_values
    max_run = 0
    current_run = 0
    for a in above:
        if a:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return max_run


def bubble_turnover(result: MarketResult, market: SSWMarket) -> float:
    """
    Turnover: total volume / (n_periods * n_shares_outstanding).

    Paper metric: measures total trading activity normalised by market size.
    Higher testosterone => higher turnover (more active trading).

    Returns:
        Turnover ratio
    """
    total_volume = result.volume.sum()
    max_volume = market.n_periods * market.n_shares
    return float(total_volume / max(max_volume, 1))


def rad(result: MarketResult) -> float:
    """
    Relative Absolute Deviation (RAD) — paper price deviation metric.

    RAD = (1/T) * sum_t |P_t - FV_t| / FV_t

    Measures average unsigned deviation from fundamental value.
    """
    T = len(result.periods)
    fv = result.fundamental_values
    prices = result.transaction_prices
    valid = fv > 0
    if not valid.any():
        return 0.0
    return float(np.mean(np.abs(prices[valid] - fv[valid]) / fv[valid]))


def rd(result: MarketResult) -> float:
    """
    Relative Deviation (RD) — signed version of RAD.

    RD = (1/T) * sum_t (P_t - FV_t) / FV_t

    Positive => overpricing (bubble), Negative => underpricing.
    """
    fv = result.fundamental_values
    prices = result.transaction_prices
    valid = fv > 0
    if not valid.any():
        return 0.0
    return float(np.mean((prices[valid] - fv[valid]) / fv[valid]))


def overbidding_rate(result: MarketResult) -> float:
    """
    Fraction of bids that exceed the fundamental value each period.

    Paper finding: testosterone-treated traders overbid more frequently.
    """
    overbid_fracs = []
    for t in range(len(result.periods)):
        fv = result.fundamental_values[t]
        if fv > 0:
            frac = float(np.mean(result.bids[t] > fv))
            overbid_fracs.append(frac)
    return float(np.mean(overbid_fracs)) if overbid_fracs else 0.0


def bubble_report(result: MarketResult, market: SSWMarket) -> dict:
    """
    Full bubble metrics report for one market simulation.

    Returns:
        Dict of all metrics matching paper Table 2
    """
    return {
        "treatment":       result.treatment,
        "amplitude":       bubble_amplitude(result),
        "duration":        bubble_duration(result),
        "turnover":        bubble_turnover(result, market),
        "RAD":             rad(result),
        "RD":              rd(result),
        "overbid_rate":    overbidding_rate(result),
        "mean_price":      float(result.transaction_prices.mean()),
        "mean_fv":         float(result.fundamental_values.mean()),
        "price_vol":       float(result.transaction_prices.std()),
    }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Bull of Wall Street — Experimental Asset Bubbles")
    print("=" * 60)

    rng = np.random.default_rng(42)
    market = SSWMarket(n_periods=15, n_traders=6)

    print(f"\n  Fundamental value schedule (T=15 periods):")
    fv = market.fundamental_value_series()
    print(f"  {fv[:8].round(2)}  ...  ({fv[-1]:.2f} at expiry)")

    # Placebo group: low testosterone (near zero)
    t_placebo = rng.normal(0.0, 0.3, market.n_traders)
    result_placebo = simulate_market(market, t_placebo, treatment="placebo", seed=42)

    # Testosterone group: elevated testosterone (positive mean)
    t_treatment = rng.normal(1.5, 0.4, market.n_traders)
    result_treatment = simulate_market(market, t_treatment, treatment="testosterone", seed=42)

    print(f"\n  {'Metric':<20} {'Placebo':>12} {'Testosterone':>14}")
    print(f"  {'-'*20} {'-'*12} {'-'*14}")

    metrics_placebo = bubble_report(result_placebo, market)
    metrics_treatment = bubble_report(result_treatment, market)

    for key in ["amplitude", "duration", "turnover", "RAD", "RD", "overbid_rate"]:
        p = metrics_placebo[key]
        t = metrics_treatment[key]
        diff = t - p
        print(f"  {key:<20} {p:>12.3f} {t:>14.3f}  (Δ={diff:+.3f})")

    print(f"\n  Paper prediction: testosterone group should show:")
    print(f"    ✓ Higher amplitude (prices further above FV)")
    print(f"    ✓ Longer duration (bubbles last longer)")
    print(f"    ✓ Higher turnover (more trading activity)")
    print(f"    ✓ Higher RAD/RD (greater price deviation)")
    print(f"    ✓ Higher overbid rate (more bids above FV)")

    # Show how T level modulates bubble amplitude
    print(f"\n  Testosterone level vs Bubble Amplitude:")
    print(f"  {'T level':>10} | {'Amplitude':>10} | {'RD':>8}")
    print(f"  {'-'*10}---{'-'*10}---{'-'*8}")
    for t_mean in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]:
        t_levels = np.full(market.n_traders, t_mean) + rng.normal(0, 0.1, market.n_traders)
        res = simulate_market(market, t_levels, seed=99)
        metrics = bubble_report(res, market)
        print(f"  {t_mean:>10.1f} | {metrics['amplitude']:>10.4f} | {metrics['RD']:>8.3f}")
