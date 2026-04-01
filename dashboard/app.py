"""
dashboard/app.py — Draculative Alpha Engine · Streamlit Dashboard

Run with:
    streamlit run dashboard/app.py

Tabs:
  1. 📊 Equity Curve & P&L
  2. 🎯 Signal Log
  3. 🔬 Concept Registry
  4. 📅 Weekly Report
  5. 🌍 Macro & News
  6. 🧠 Nucleus State
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Draculative Alpha Engine",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🧬 Draculative Alpha Engine")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["📊 Equity Curve", "🎯 Signal Log", "🔬 Concept Registry",
     "📅 Weekly Report", "🌍 Macro & News", "🧠 Nucleus State"],
)
st.sidebar.markdown("---")
st.sidebar.caption("Version: Alpha 1.0 · Atom-Nucleus Hypothesis")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Equity Curve
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Equity Curve":
    st.title("📊 Equity Curve & Performance Metrics")

    # Look for any SimulationResult JSON in reports/
    sim_files = sorted(Path("reports").rglob("*.json")) if Path("reports").exists() else []
    if not sim_files:
        st.info("No simulation results found. Run a simulation first.")
    else:
        sel_file = st.selectbox("Select simulation result", sim_files, format_func=lambda p: p.name)
        try:
            with open(sel_file) as f:
                data = json.load(f)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Profit Factor",   f"{data.get('profit_factor', 0):.2f}")
            col2.metric("Win Rate",        f"{data.get('win_rate', 0)*100:.1f}%")
            col3.metric("Sharpe",          f"{data.get('sharpe_annualized', 0):.2f}")
            col4.metric("Max Drawdown",    f"{data.get('max_drawdown_pct', 0)*100:.1f}%")

            col5, col6, col7 = st.columns(3)
            col5.metric("Total Trades",    data.get("total_trades", 0))
            col6.metric("Calmar Ratio",    f"{data.get('calmar_ratio', 0):.2f}")
            col7.metric("Trades / Month",  f"{data.get('trades_per_month', 0):.1f}")

            # Monthly breakdown
            mb = data.get("monthly_breakdown", [])
            if mb:
                df_mb = pd.DataFrame(mb)
                df_mb["wr%"] = df_mb["wins"] / df_mb["trades"].replace(0, 1) * 100
                st.subheader("Monthly Breakdown")
                st.bar_chart(df_mb.set_index("month")["pnl"])
                st.dataframe(df_mb, use_container_width=True)
        except Exception as e:
            st.error(f"Could not load file: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Signal Log
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Signal Log":
    st.title("🎯 Signal Log")

    log_path = Path("data/signal_log.csv")
    if log_path.exists():
        df = pd.read_csv(log_path, parse_dates=["entry_time", "exit_time"])
        st.dataframe(df, use_container_width=True)

        # P&L distribution
        if "pnl" in df.columns:
            st.subheader("P&L Distribution")
            st.bar_chart(df["pnl"].value_counts().sort_index())

        # Win rate per concept
        if "concept" in df.columns:
            st.subheader("Win Rate by Concept")
            wr = df.groupby("concept").apply(
                lambda g: (g["pnl"] > 0).mean()
            ).reset_index(name="win_rate")
            st.bar_chart(wr.set_index("concept")["win_rate"])
    else:
        st.info("No signal log found at data/signal_log.csv. Run a simulation to populate it.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Concept Registry
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Concept Registry":
    st.title("🔬 Concept Registry")

    try:
        from technical.concept_registry import CONCEPT_REGISTRY, load_active_concepts
        rows = []
        for entry in CONCEPT_REGISTRY:
            rows.append({
                "Concept":    entry["name"],
                "Category":   entry.get("category", ""),
                "Source":     entry.get("source", ""),
                "Version":    entry.get("version", ""),
                "Status":     entry["status"].value if hasattr(entry["status"], "value") else str(entry["status"]),
                "WFE":        entry.get("wfe", "-"),
                "PF (test)":  entry.get("pf_test", "-"),
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        st.caption(f"Total: {len(rows)} concepts registered")
    except Exception as e:
        st.error(f"Could not load registry: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Weekly Report
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📅 Weekly Report":
    st.title("📅 Weekly Report")

    report_files = sorted(Path("reports/weekly").glob("*_weekly_report.txt")) \
        if Path("reports/weekly").exists() else []
    if not report_files:
        st.info("No weekly reports found. Run the weekly report generator (Friday 6pm ET).")
    else:
        sel = st.selectbox("Select report", report_files, format_func=lambda p: p.name)
        st.text(sel.read_text())

    if st.button("🔄 Generate Report Now"):
        try:
            from fundamental.weekly_report import WeeklyReportGenerator
            gen = WeeklyReportGenerator()
            rpt = gen.generate()
            st.success(f"Report generated: {rpt.week_ending}")
            st.text(Path(f"reports/weekly/{rpt.week_ending}_weekly_report.txt").read_text())
        except Exception as e:
            st.error(f"Report generation failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Macro & News
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🌍 Macro & News":
    st.title("🌍 Macro & Upcoming Events")

    try:
        from fundamental.macro_tracker import MacroTracker
        mt    = MacroTracker()
        state = mt.get_state()
        col1, col2, col3 = st.columns(3)
        col1.metric("Macro Regime", state.regime.value)
        col2.metric("VIX",          f"{state.vix:.1f}")
        col3.metric("SPY vs SMA200", f"{state.spy_vs_200*100:+.1f}%")
    except Exception as e:
        st.warning(f"Macro data unavailable: {e}")

    st.markdown("---")
    st.subheader("📅 Upcoming High-Impact Events (7 days)")
    try:
        from fundamental.news_fetcher import NewsFetcher
        nf      = NewsFetcher()
        events  = nf.get_high_impact(days_ahead=7)
        if events:
            rows = [{"Date": e.date, "Time (ET)": e.time_et,
                     "Title": e.title, "Category": e.category,
                     "Impact": e.impact} for e in events]
            st.table(pd.DataFrame(rows))
        else:
            st.info("No high-impact events in the next 7 days (static calendar).")
    except Exception as e:
        st.warning(f"News data unavailable: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — Nucleus State
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 Nucleus State":
    st.title("🧠 Nucleus Identification Engine")
    st.info(
        "The Nucleus Engine scores 8 candidate structures on every bar and returns "
        "the dominant one. Connect live Alpaca feed to see real-time nucleus state."
    )

    nucleus_dist_path = Path("data/nucleus_log.json")
    if nucleus_dist_path.exists():
        try:
            with open(nucleus_dist_path) as f:
                dist = json.load(f)
            st.subheader("Nucleus Frequency Distribution (all bars)")
            df_n = pd.DataFrame(list(dist.items()), columns=["Nucleus", "Count"])
            df_n = df_n.sort_values("Count", ascending=False)
            st.bar_chart(df_n.set_index("Nucleus")["Count"])
            st.dataframe(df_n, use_container_width=True)
        except Exception as e:
            st.error(f"Could not load nucleus log: {e}")
    else:
        st.info("No nucleus log found yet. Run a simulation to populate data/nucleus_log.json.")
