"""
Research Implementations Runner
================================
Runs all 6 paper implementations in sequence and prints results.

Usage:
    cd /Users/parthvijayvargiya/Documents/GitHub/draculative
    python research/run_research.py
"""

import sys
import traceback

SEPARATOR = "\n" + "=" * 70 + "\n"


def run_module(name: str, module_path: str):
    """Import and run a module's __main__ block by executing it as __main__."""
    print(SEPARATOR)
    print(f"  ▶  {name}")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("__main__", module_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        print(f"\n  ✅ {name} — OK")
    except SystemExit:
        pass
    except Exception as e:
        print(f"  ❌ FAILED: {name}")
        print(f"     {type(e).__name__}: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    import os
    base = os.path.join(os.path.dirname(__file__), "implementations")

    modules = [
        ("01 — Relativistic Black-Scholes",       "01_relativistic_black_scholes.py"),
        ("02 — Mean-Field Game Trading",           "02_mean_field_game_trading.py"),
        ("03 — Wasserstein Regime Detection",      "03_wasserstein_regime_detection.py"),
        ("04 — ML Multi-Factor Trading",           "04_ml_multifactor_trading.py"),
        ("05 — FCOC Volatility Forecasting",       "05_fcoc_volatility.py"),
        ("06 — Experimental Bubble Metrics",       "06_bubble_metrics.py"),
    ]

    print("\n" + "=" * 70)
    print("  RESEARCH PAPER IMPLEMENTATIONS — DEMO RUNNER")
    print("=" * 70)
    print("\nRunning all 6 implementations...\n")

    for name, filename in modules:
        run_module(name, os.path.join(base, filename))

    print(SEPARATOR)
    print("  ✅ All implementations completed.")
    print("=" * 70)
