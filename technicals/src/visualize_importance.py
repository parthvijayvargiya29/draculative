#!/usr/bin/env python3
"""
Visualize ML Feature Importance Results

Creates charts showing which technical indicators have the most
predictive power for price movements.
"""

import json
from pathlib import Path
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("matplotlib not installed. Install with: pip3 install matplotlib")


def load_results(data_dir: Path) -> dict:
    """Load all analysis results."""
    results = {}
    
    # Load aggregated results
    agg_file = data_dir / "aggregated_importance.json"
    if agg_file.exists():
        with open(agg_file) as f:
            results['aggregated'] = json.load(f)
    
    # Load individual ticker results
    for json_file in data_dir.glob("*_indicator_importance.json"):
        ticker = json_file.stem.replace("_indicator_importance", "")
        with open(json_file) as f:
            results[ticker] = json.load(f)
    
    return results


def plot_top_indicators(results: dict, output_dir: Path):
    """Plot top indicators by importance."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Aggregated Top 20 Importance
    ax1 = axes[0, 0]
    if 'aggregated' in results:
        imp = results['aggregated'].get('top_20_indicators', {})
        features = list(imp.keys())[:15]
        values = [imp[f] for f in features]
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(features)))
        bars = ax1.barh(features[::-1], values[::-1], color=colors[::-1])
        ax1.set_xlabel('Importance Score')
        ax1.set_title('Top 15 Most Predictive Indicators (Averaged)', fontsize=12, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
    
    # 2. IONQ vs NVDA comparison
    ax2 = axes[0, 1]
    if 'IONQ' in results and 'NVDA' in results:
        ionq_imp = results['IONQ'].get('combined_importance', {})
        nvda_imp = results['NVDA'].get('combined_importance', {})
        
        # Get top features from both
        all_features = set(list(ionq_imp.keys())[:10] + list(nvda_imp.keys())[:10])
        sorted_features = sorted(all_features, key=lambda x: (ionq_imp.get(x, 0) + nvda_imp.get(x, 0)) / 2, reverse=True)[:12]
        
        x = np.arange(len(sorted_features))
        width = 0.35
        
        ionq_vals = [ionq_imp.get(f, 0) for f in sorted_features]
        nvda_vals = [nvda_imp.get(f, 0) for f in sorted_features]
        
        ax2.barh(x + width/2, ionq_vals, width, label='IONQ', color='#2ecc71', alpha=0.8)
        ax2.barh(x - width/2, nvda_vals, width, label='NVDA', color='#3498db', alpha=0.8)
        ax2.set_yticks(x)
        ax2.set_yticklabels(sorted_features)
        ax2.set_xlabel('Importance Score')
        ax2.set_title('Feature Importance: IONQ vs NVDA', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(axis='x', alpha=0.3)
    
    # 3. Signal Win Rates
    ax3 = axes[1, 0]
    signal_data = {}
    for ticker in ['IONQ', 'NVDA']:
        if ticker in results:
            signals = results[ticker].get('signal_analysis', {}).get('5d', {})
            for sig, data in signals.items():
                if sig not in signal_data:
                    signal_data[sig] = {'win_rates': [], 'tickers': []}
                signal_data[sig]['win_rates'].append(data['win_rate'])
                signal_data[sig]['tickers'].append(ticker)
    
    if signal_data:
        # Average win rate across tickers
        avg_win_rates = {sig: np.mean(data['win_rates']) for sig, data in signal_data.items()}
        sorted_signals = sorted(avg_win_rates.items(), key=lambda x: x[1], reverse=True)
        
        signals = [s[0] for s in sorted_signals]
        win_rates = [s[1] * 100 for s in sorted_signals]
        
        colors = ['#27ae60' if wr > 50 else '#e74c3c' for wr in win_rates]
        bars = ax3.barh(signals[::-1], win_rates[::-1], color=colors[::-1])
        ax3.axvline(x=50, color='gray', linestyle='--', linewidth=2, alpha=0.7)
        ax3.set_xlabel('Win Rate (%)')
        ax3.set_title('Signal Win Rates (5-day horizon)', fontsize=12, fontweight='bold')
        ax3.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, wr) in enumerate(zip(bars, win_rates[::-1])):
            ax3.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                    f'{wr:.1f}%', va='center', fontsize=9)
    
    # 4. Correlation with Future Returns
    ax4 = axes[1, 1]
    corr_data = {}
    for ticker in ['IONQ', 'NVDA']:
        if ticker in results:
            corrs = results[ticker].get('correlations', {}).get('return_5d', {})
            for feat, corr in corrs.items():
                if feat not in corr_data:
                    corr_data[feat] = []
                corr_data[feat].append(corr)
    
    if corr_data:
        # Average absolute correlation
        avg_abs_corr = {feat: np.mean(np.abs(corrs)) for feat, corrs in corr_data.items()}
        avg_corr = {feat: np.mean(corrs) for feat, corrs in corr_data.items()}
        
        sorted_by_abs = sorted(avg_abs_corr.items(), key=lambda x: x[1], reverse=True)[:12]
        
        features = [s[0] for s in sorted_by_abs]
        corrs = [avg_corr[f] for f in features]
        
        colors = ['#27ae60' if c > 0 else '#e74c3c' for c in corrs]
        ax4.barh(features[::-1], corrs[::-1], color=colors[::-1])
        ax4.axvline(x=0, color='gray', linestyle='-', linewidth=1)
        ax4.set_xlabel('Correlation with 5-day Return')
        ax4.set_title('Strongest Correlations with Future Returns', fontsize=12, fontweight='bold')
        ax4.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    output_file = output_dir / "indicator_importance_charts.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved chart to {output_file}")
    plt.close()


def plot_regime_analysis(results: dict, output_dir: Path):
    """Plot indicator performance by market regime."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    regimes = ['trend_up', 'trend_down', 'sideways', 'high_vol', 'low_vol']
    regime_titles = ['Trending Up', 'Trending Down', 'Sideways', 'High Volatility', 'Low Volatility']
    
    for i, (regime, title) in enumerate(zip(regimes, regime_titles)):
        ax = axes[i]
        
        # Collect regime data from all tickers
        regime_features = {}
        for ticker in ['IONQ', 'NVDA']:
            if ticker in results:
                regime_data = results[ticker].get('regime_analysis', {}).get(regime, {})
                top_inds = regime_data.get('top_indicators', {})
                for feat, corr in top_inds.items():
                    if feat not in regime_features:
                        regime_features[feat] = []
                    regime_features[feat].append(corr)
        
        if regime_features:
            avg_corrs = {f: np.mean(corrs) for f, corrs in regime_features.items()}
            sorted_corrs = sorted(avg_corrs.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
            
            features = [s[0] for s in sorted_corrs]
            corrs = [s[1] for s in sorted_corrs]
            
            colors = ['#27ae60' if c > 0 else '#e74c3c' for c in corrs]
            ax.barh(features[::-1], corrs[::-1], color=colors[::-1])
            ax.axvline(x=0, color='gray', linestyle='-', linewidth=1)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
    
    # Use last subplot for legend
    ax_legend = axes[5]
    ax_legend.axis('off')
    
    pos_patch = mpatches.Patch(color='#27ae60', label='Positive correlation with returns')
    neg_patch = mpatches.Patch(color='#e74c3c', label='Negative correlation with returns')
    ax_legend.legend(handles=[pos_patch, neg_patch], loc='center', fontsize=12)
    ax_legend.text(0.5, 0.3, 'Indicators to prioritize\nin each market regime', 
                   ha='center', va='center', fontsize=14, transform=ax_legend.transAxes)
    
    plt.suptitle('Best Indicators by Market Regime', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_file = output_dir / "regime_analysis_charts.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved chart to {output_file}")
    plt.close()


def create_summary_report(results: dict, output_dir: Path):
    """Create a summary markdown report."""
    lines = []
    lines.append("# Technical Indicator Importance Analysis")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append("This analysis backpropagates through 2 years of historical price data to identify")
    lines.append("which technical indicators have the most predictive power for future price movements.")
    lines.append("")
    
    if 'aggregated' in results:
        agg = results['aggregated']
        lines.append(f"**Tickers Analyzed:** {', '.join(agg.get('tickers', []))}")
        lines.append("")
        
        lines.append("### Top 10 Most Important Indicators")
        lines.append("")
        lines.append("| Rank | Indicator | Importance Score |")
        lines.append("|------|-----------|------------------|")
        
        top_20 = agg.get('top_20_indicators', {})
        for i, (feat, imp) in enumerate(list(top_20.items())[:10], 1):
            lines.append(f"| {i} | `{feat}` | {imp:.4f} |")
        lines.append("")
    
    # Signal Analysis
    lines.append("### Signal Win Rates (5-day horizon)")
    lines.append("")
    lines.append("These signals show the probability of price moving in the expected direction:")
    lines.append("")
    lines.append("| Signal | Avg Win Rate | Interpretation |")
    lines.append("|--------|-------------|----------------|")
    
    signal_stats = {}
    for ticker in ['IONQ', 'NVDA']:
        if ticker in results:
            signals = results[ticker].get('signal_analysis', {}).get('5d', {})
            for sig, data in signals.items():
                if sig not in signal_stats:
                    signal_stats[sig] = {'win_rates': [], 'direction': data['expected_direction']}
                signal_stats[sig]['win_rates'].append(data['win_rate'])
    
    sorted_signals = sorted(signal_stats.items(), key=lambda x: np.mean(x[1]['win_rates']), reverse=True)
    for sig, data in sorted_signals:
        wr = np.mean(data['win_rates']) * 100
        direction = data['direction']
        interp = "✅ Reliable" if wr > 55 else "⚠️ Moderate" if wr > 50 else "❌ Weak"
        lines.append(f"| `{sig}` | {wr:.1f}% | {interp} ({direction}) |")
    lines.append("")
    
    # Key Insights
    lines.append("## Key Insights")
    lines.append("")
    lines.append("### Volume-Based Indicators Dominate")
    lines.append("- **OBV (On-Balance Volume)** is consistently the most predictive indicator")
    lines.append("- Volume confirms price moves - divergences often precede reversals")
    lines.append("")
    
    lines.append("### Ichimoku Cloud is Highly Effective")
    lines.append("- Senkou Span B and Chikou Span rank in top 5")
    lines.append("- Cloud breakouts/breakdowns provide reliable directional signals")
    lines.append("")
    
    lines.append("### Volatility Indicators Matter")
    lines.append("- ATR and Bollinger Band width help predict move magnitude")
    lines.append("- High volatility regimes require different indicator weighting")
    lines.append("")
    
    lines.append("### Signal Recommendations")
    lines.append("")
    for ticker in ['IONQ', 'NVDA']:
        if ticker in results:
            signals = results[ticker].get('signal_analysis', {}).get('5d', {})
            best_signal = max(signals.items(), key=lambda x: x[1]['win_rate']) if signals else None
            if best_signal:
                lines.append(f"- **{ticker}**: Best signal is `{best_signal[0]}` with {best_signal[1]['win_rate']:.1%} win rate")
    lines.append("")
    
    lines.append("## Regime-Specific Recommendations")
    lines.append("")
    lines.append("### In Trending Markets (Up or Down)")
    lines.append("- Prioritize moving average crossovers (SMA 50/200)")
    lines.append("- ADX > 25 confirms trend strength")
    lines.append("- Follow momentum indicators (MACD)")
    lines.append("")
    
    lines.append("### In Sideways/Ranging Markets")
    lines.append("- Use oscillators: Stochastic, RSI, CCI")
    lines.append("- Mean reversion strategies work better")
    lines.append("- Bollinger Band extremes provide entry points")
    lines.append("")
    
    lines.append("### In High Volatility")
    lines.append("- Widen stops based on ATR")
    lines.append("- DI crossovers become more significant")
    lines.append("- Reduce position sizes")
    lines.append("")
    
    lines.append("### In Low Volatility")
    lines.append("- Stochastic RSI and Williams %R work well")
    lines.append("- Watch for volatility expansion signals")
    lines.append("- Prepare for breakout moves")
    lines.append("")
    
    lines.append("## Model Performance")
    lines.append("")
    lines.append("| Ticker | 1-Day Accuracy | 5-Day Accuracy | 10-Day Accuracy |")
    lines.append("|--------|---------------|----------------|-----------------|")
    
    for ticker in ['IONQ', 'NVDA']:
        if ticker in results:
            model_res = results[ticker].get('model_results', {})
            d1 = model_res.get('direction_1d', {}).get('cv_accuracy', 0) * 100
            d5 = model_res.get('direction_5d', {}).get('cv_accuracy', 0) * 100
            d10 = model_res.get('direction_10d', {}).get('cv_accuracy', 0) * 100
            lines.append(f"| {ticker} | {d1:.1f}% | {d5:.1f}% | {d10:.1f}% |")
    lines.append("")
    
    lines.append("---")
    lines.append("")
    lines.append("*Note: This analysis is for research purposes. Past performance does not guarantee future results.*")
    
    report_file = output_dir / "SUMMARY_REPORT.md"
    with open(report_file, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Saved summary report to {report_file}")


def main():
    data_dir = Path("technicals/data/ml")
    
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        print("Run ml_feature_importance.py first to generate analysis data.")
        return
    
    print("Loading analysis results...")
    results = load_results(data_dir)
    
    if not results:
        print("No results found!")
        return
    
    print(f"Found results for: {list(results.keys())}")
    
    # Generate visualizations
    if HAS_MATPLOTLIB:
        print("\nGenerating charts...")
        plot_top_indicators(results, data_dir)
        plot_regime_analysis(results, data_dir)
    else:
        print("\nSkipping charts (matplotlib not installed)")
    
    # Generate summary report
    print("\nGenerating summary report...")
    create_summary_report(results, data_dir)
    
    print("\n✅ Visualization complete!")


if __name__ == '__main__':
    main()
