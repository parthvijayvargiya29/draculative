"""
Streamlit Dashboard for Draculative Stock Predictor

Run:
  pip install -r predictor/app/requirements.txt
  streamlit run predictor/app/dashboard.py

This app imports the existing `StockPredictor` to generate live predictions,
plots a candlestick chart with support/resistance/targets, and shows a clear
explanation of signals and action items.
"""

import sys
from datetime import datetime
from typing import List

# Ensure we can import from predictor/src
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import yfinance as yf

try:
    from stock_predictor import StockPredictor
except ImportError:
    # Fallback: manually add predictor/src to path
    import sys
    predictor_src = os.path.join(ROOT, 'src')
    if predictor_src not in sys.path:
        sys.path.insert(0, predictor_src)
    from stock_predictor import StockPredictor


def fetch_prediction_and_price(ticker: str):
    """Fetch prediction using StockPredictor and return price history for charting."""
    sp = StockPredictor(ticker)
    pred = sp.generate_prediction()

    # Fetch OHLCV for plotting (90 days)
    df = yf.Ticker(ticker).history(period='90d', interval='1d').reset_index()
    return pred, df


def plot_price_with_levels(df, pred):
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))

    # Horizontal lines for support/resistance/targets/stop
    price = pred.current_price
    support = pred.technical.support
    resistance = pred.technical.resistance
    stop = pred.stop_loss
    t1 = pred.target_price_bull
    t2 = pred.target_price_bear

    fig.add_hline(y=price, line=dict(color='gray', dash='dash'), annotation_text='Current', annotation_position='top left')
    fig.add_hline(y=support, line=dict(color='green'), annotation_text=f'Support {support}', annotation_position='bottom left')
    fig.add_hline(y=resistance, line=dict(color='red'), annotation_text=f'Resistance {resistance}', annotation_position='top right')
    fig.add_hline(y=stop, line=dict(color='orange', dash='dot'), annotation_text=f'Stop {stop}', annotation_position='bottom right')
    fig.add_hline(y=t1, line=dict(color='blue', dash='dash'), annotation_text=f'Target Bull {t1}', annotation_position='top left')
    fig.add_hline(y=t2, line=dict(color='purple', dash='dash'), annotation_text=f'Target Bear {t2}', annotation_position='bottom left')

    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=450)
    return fig


def explain_prediction(pred):
    """Produce a textual explanation describing technical/fundamental/news reasons."""
    lines: List[str] = []
    lines.append(f"Ticker: {pred.ticker} — {pred.direction} ({pred.confidence:.0%} confidence)")
    lines.append(f"Price: ${pred.current_price:.2f} | Volatility (ATR%): {pred.volatility}% | R/R: {pred.risk_reward_ratio}")
    lines.append("")
    lines.append("Technical summary:")
    lines.append(f"  - Direction: {pred.technical.direction} (strength={pred.technical.strength}, conf={pred.technical.confidence})")
    if pred.technical.bullish_factors:
        lines.append(f"  - Bullish: {pred.technical.bullish_factors[0]}")
    if pred.technical.bearish_factors:
        lines.append(f"  - Bearish: {pred.technical.bearish_factors[0]}")
    lines.append("")
    lines.append("Fundamental summary:")
    lines.append(f"  - Direction: {pred.fundamental.direction} (strength={pred.fundamental.strength}, conf={pred.fundamental.confidence})")
    if pred.fundamental.bullish_factors:
        lines.append(f"  - Bullish: {pred.fundamental.bullish_factors[0]}")
    if pred.fundamental.bearish_factors:
        lines.append(f"  - Bearish: {pred.fundamental.bearish_factors[0]}")
    lines.append("")
    lines.append("News / events summary:")
    lines.append(f"  - Direction: {pred.news.direction} (strength={pred.news.strength}, conf={pred.news.confidence})")
    if pred.news.detected_events:
        lines.append(f"  - Events: {', '.join(pred.news.detected_events[:3])}")

    lines.append("")
    lines.append("Price targets & trade plan:")
    lines.append(f"  - Entry signal: {pred.direction}")
    lines.append(f"  - Support: ${pred.technical.support:.2f}")
    lines.append(f"  - Resistance: ${pred.technical.resistance:.2f}")
    lines.append(f"  - Stop Loss: ${pred.stop_loss:.2f}")
    lines.append(f"  - Target (bull): ${pred.target_price_bull:.2f}")
    lines.append(f"  - Target (bear): ${pred.target_price_bear:.2f}")
    lines.append("")
    lines.append("Action items:")
    for a in pred.action_items:
        lines.append(f"  - {a}")

    return "\n".join(lines)


def main():
    import streamlit as st

    st.set_page_config(page_title="Draculative — Live Stock Predictor Dashboard", layout="wide")
    st.title("Draculative — Live Stock Predictor Dashboard")

    with st.sidebar:
        st.header("Controls")
        tickers = st.text_input("Tickers (comma-separated)", value="NVDA, IONQ")
        ticker_list = [t.strip().upper() for t in tickers.split(',') if t.strip()]
        selected = st.selectbox("Select ticker", ticker_list)
        run = st.button("Run Prediction")

    if run and selected:
        with st.spinner(f"Generating prediction for {selected}..."):
            pred, price_df = fetch_prediction_and_price(selected)

        # Layout
        left, right = st.columns([2, 1])

        with left:
            st.subheader(f"{selected} — {pred.direction} ({pred.confidence:.0%})")
            fig = plot_price_with_levels(price_df, pred)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Technical Explanation")
            st.code(explain_prediction(pred))

        with right:
            st.subheader("Quick Stats")
            st.metric("Price", f"${pred.current_price:.2f}")
            st.metric("Direction", pred.direction)
            st.metric("Confidence", f"{pred.confidence:.0%}")
            st.markdown("---")
            st.subheader("Support / Resistance / Targets")
            st.write(f"Support: ${pred.technical.support:.2f}")
            st.write(f"Resistance: ${pred.technical.resistance:.2f}")
            st.write(f"Stop Loss: ${pred.stop_loss:.2f}")
            st.write(f"Target (Bull): ${pred.target_price_bull:.2f}")
            st.write(f"Risk/Reward: {pred.risk_reward_ratio}")

    else:
        st.info("Enter tickers and press 'Run Prediction' to fetch live data and see the dashboard.")


# Helper for CLI smoke-test
def run_test():
    pred, _ = fetch_prediction_and_price('NVDA')
    print('=== SMOKE TEST ===')
    print(pred.summary)


if __name__ == '__main__':
    main()
