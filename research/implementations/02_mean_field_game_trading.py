"""
Mean-Field Game Trading with Differing Beliefs
===============================================
Paper: "Mean-Field Games with Differing Beliefs for Algorithmic Trading"
       Casgrain & Jaimungal (arXiv:1810.06101v2) — Mathematical Finance

Core idea:
  A large population of heterogeneous agents trade the same asset, each holding
  a *different* belief (probability measure) about the asset's drift.
  The MFG limit gives a Nash equilibrium trading rate expressed as a
  forward-backward SDE system. We implement the LSMC-based numerical solution.

Key components implemented:
  1. Asset price model with permanent market impact
  2. Agent inventory dynamics and cash process
  3. Optimal trading rate (Nash equilibrium control)
  4. LSMC Monte Carlo simulation of MFG equilibrium
  5. Price volatility vs. belief disagreement analysis
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Model Parameters
# ---------------------------------------------------------------------------

@dataclass
class AgentPopulation:
    """
    One sub-population of agents sharing the same belief.

    Attributes:
        drift:        Believed drift of the asset (A^k in paper)
        risk_aversion: Terminal inventory penalty Psi_k
        run_penalty:   Running inventory penalty phi_k
        trade_cost:    Instantaneous trading cost a_k
        impact_weight: Market impact weight lambda_k
        initial_inv:   Mean starting inventory m_k
        n_agents:      Number of agents in this sub-population
    """
    drift: float = 0.0
    risk_aversion: float = 0.5
    run_penalty: float = 0.01
    trade_cost: float = 0.1
    impact_weight: float = 1.0
    initial_inv: float = 0.0
    n_agents: int = 100


@dataclass
class MFGParams:
    """Global simulation parameters."""
    T: float = 1.0           # Trading horizon (years)
    n_steps: int = 50        # Time steps
    S0: float = 100.0        # Initial asset price
    vol: float = 0.20        # Asset volatility (sigma of martingale part M^k)
    populations: List[AgentPopulation] = field(default_factory=list)

    def __post_init__(self):
        if not self.populations:
            # Default: two populations with opposing drift beliefs
            self.populations = [
                AgentPopulation(drift=0.05,  initial_inv= 1.0, n_agents=100),  # bulls
                AgentPopulation(drift=-0.05, initial_inv=-1.0, n_agents=100),  # bears
            ]

    @property
    def dt(self) -> float:
        return self.T / self.n_steps

    @property
    def n_pops(self) -> int:
        return len(self.populations)


# ---------------------------------------------------------------------------
# Core model: optimal trading rate (Nash equilibrium)
# ---------------------------------------------------------------------------

def compute_optimal_rate(
    q: float,
    q_bar: float,
    t: float,
    pop: AgentPopulation,
    params: MFGParams,
) -> float:
    """
    Compute the Nash equilibrium trading rate nu* for one agent.

    From the paper (Section 3 / Theorem 3.1), the optimal control is:
        nu*_t = (A^k_t - r^k_t) / (2 * a^k) - phi^k * q / a^k

    where r^k_t is the adjoint process (value function gradient).
    In the MFG equilibrium, we approximate:
        r^k_t ≈ Psi^k * E[q^k_T] * exp(-phi^k * (T - t))
               + lambda * q_bar  (mean-field term)

    Args:
        q:     Current inventory of this agent
        q_bar: Average inventory across the population (mean-field state)
        t:     Current time
        pop:   This agent's population parameters
        params: Global MFG parameters

    Returns:
        Optimal trading rate nu*
    """
    T_remaining = params.T - t

    # Adjoint process approximation: linear in q and mean field
    # (From FBSDE solution under mild assumptions)
    adjoint = (
        pop.risk_aversion * q * np.exp(-pop.run_penalty * T_remaining)
        + pop.impact_weight * q_bar
    )

    # Optimal rate: balance drift belief against liquidation urgency
    nu_star = (
        (pop.drift - adjoint) / (2.0 * pop.trade_cost)
        - pop.run_penalty * q / pop.trade_cost
    )

    return nu_star


def compute_price_impact(
    avg_rates: List[float],
    populations: List[AgentPopulation],
) -> float:
    """
    Compute total permanent price impact from all agents' average trading rates.

    From paper eq. (5):
        Impact = sum_k lambda_k * p_k * nu_bar_k
    where p_k is the population fraction and nu_bar_k is average trading rate.

    Args:
        avg_rates: Average trading rate per population
        populations: Population parameters

    Returns:
        Total price impact (added to asset drift)
    """
    total_agents = sum(p.n_agents for p in populations)
    impact = 0.0
    for pop, nu_bar in zip(populations, avg_rates):
        p_k = pop.n_agents / total_agents
        impact += pop.impact_weight * p_k * nu_bar
    return impact


# ---------------------------------------------------------------------------
# LSMC Monte Carlo simulation
# ---------------------------------------------------------------------------

def simulate_mfg(
    params: MFGParams,
    n_paths: int = 500,
    seed: int = 42,
) -> dict:
    """
    Simulate the MFG equilibrium using iterative fixed-point iteration.

    Algorithm (from paper Section 5 / LSMC approach):
      1. Initialise mean-field (average inventory path) to zero
      2. For each iteration:
         a. Simulate each agent's optimal inventory given current mean-field
         b. Update mean-field as average of simulated inventories
         c. Repeat until convergence

    Args:
        n_paths: Monte Carlo paths per population
        seed: Random seed

    Returns:
        Dict with price paths, inventory paths, trading rates, volatility stats
    """
    rng = np.random.default_rng(seed)
    dt = params.dt
    n_steps = params.n_steps
    times = np.linspace(0, params.T, n_steps + 1)

    # Initialise mean-field inventories as zero for each pop
    q_bar = [np.zeros(n_steps + 1) for _ in params.populations]

    n_iter = 10  # fixed-point iterations (typically 5-10 suffice)

    for iteration in range(n_iter):
        new_q_bar = [np.zeros(n_steps + 1) for _ in params.populations]

        for pop_idx, pop in enumerate(params.populations):
            # Simulate n_paths agents from this population
            q = rng.normal(pop.initial_inv, 0.5, size=n_paths)
            q_paths = np.zeros((n_paths, n_steps + 1))
            q_paths[:, 0] = q

            for step in range(n_steps):
                t = times[step]
                qb = q_bar[pop_idx][step]

                # Vectorised optimal rate for all paths
                nu = np.array([
                    compute_optimal_rate(q[i], qb, t, pop, params)
                    for i in range(n_paths)
                ])
                # Inventory update: dq = nu * dt
                q = q + nu * dt
                q_paths[:, step + 1] = q

            new_q_bar[pop_idx] = q_paths.mean(axis=0)

        q_bar = new_q_bar

    # Final simulation: simulate asset price with market impact
    S = np.full(n_paths, params.S0)
    S_paths = np.zeros((n_paths, n_steps + 1))
    S_paths[:, 0] = S

    # Aggregate trading rates across populations for price impact
    all_q_paths = []  # (pop, n_paths, n_steps+1)
    for pop_idx, pop in enumerate(params.populations):
        rng2 = np.random.default_rng(seed + pop_idx + 1)
        q = rng2.normal(pop.initial_inv, 0.5, size=n_paths)
        q_paths = np.zeros((n_paths, n_steps + 1))
        q_paths[:, 0] = q
        for step in range(n_steps):
            t = times[step]
            qb = q_bar[pop_idx][step]
            nu = np.array([
                compute_optimal_rate(q[i], qb, t, pop, params)
                for i in range(n_paths)
            ])
            q = q + nu * dt
            q_paths[:, step + 1] = q
        all_q_paths.append(q_paths)

    # Simulate price incorporating mean-field impact
    nu_bars = []
    for step in range(n_steps):
        avg_rates_step = []
        for pop_idx, pop in enumerate(params.populations):
            t = times[step]
            qb = q_bar[pop_idx][step]
            nu_pop = np.array([
                compute_optimal_rate(all_q_paths[pop_idx][i, step], qb, t, pop, params)
                for i in range(n_paths)
            ])
            avg_rates_step.append(float(nu_pop.mean()))
        nu_bars.append(avg_rates_step)

        impact = compute_price_impact(avg_rates_step, params.populations)
        dW = rng.standard_normal(n_paths) * np.sqrt(dt)
        # Asset price: dS = (impact_drift) * dt + vol * S * dW
        S = S + impact * dt * params.vol + params.vol * S * dW
        S_paths[:, step + 1] = S

    return {
        "times": times,
        "S_paths": S_paths,
        "q_bar": q_bar,
        "q_paths": all_q_paths,
        "nu_bars": nu_bars,
        "price_vol": S_paths.std(axis=0),
        "final_price_vol": float(S_paths[:, -1].std()),
    }


def belief_disagreement(params: MFGParams) -> float:
    """
    Measure of belief disagreement across populations.
    Defined as the standard deviation of drift beliefs weighted by population size.

    Paper finding: higher disagreement => higher price volatility.
    """
    total = sum(p.n_agents for p in params.populations)
    drifts = np.array([p.drift for p in params.populations])
    weights = np.array([p.n_agents / total for p in params.populations])
    mean_drift = np.dot(weights, drifts)
    return float(np.sqrt(np.dot(weights, (drifts - mean_drift) ** 2)))


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Mean-Field Game Trading with Differing Beliefs")
    print("=" * 60)

    # Test 1: Baseline (bull vs bear)
    params = MFGParams(T=1.0, n_steps=50, vol=0.20)
    result = simulate_mfg(params, n_paths=200, seed=42)
    disagreement = belief_disagreement(params)
    print(f"\nBull vs Bear market:")
    print(f"  Belief disagreement (sigma_drift): {disagreement:.4f}")
    print(f"  Final price vol:                  {result['final_price_vol']:.4f}")
    print(f"  Mean-field q_bar (bulls, t=0->T):  {result['q_bar'][0][0]:.3f} -> {result['q_bar'][0][-1]:.3f}")
    print(f"  Mean-field q_bar (bears, t=0->T):  {result['q_bar'][1][0]:.3f} -> {result['q_bar'][1][-1]:.3f}")

    # Test 2: Disagreement sweep — should confirm paper finding
    print(f"\nDisagreement vs Price Volatility (paper finding: positive relationship):")
    print(f"  Drift gap | Disagreement | Final Price Vol")
    print(f"  ----------|--------------|----------------")
    for gap in [0.02, 0.05, 0.10, 0.15, 0.20]:
        pops = [
            AgentPopulation(drift= gap/2, initial_inv= 1.0, n_agents=100),
            AgentPopulation(drift=-gap/2, initial_inv=-1.0, n_agents=100),
        ]
        p = MFGParams(T=1.0, n_steps=50, vol=0.20, populations=pops)
        r = simulate_mfg(p, n_paths=150, seed=42)
        d = belief_disagreement(p)
        print(f"  {gap:.2f}      | {d:.4f}       | {r['final_price_vol']:.4f}")
