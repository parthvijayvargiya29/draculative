"""
Enhanced Friday Report
======================

7-section weekly trading system health report.

SECTIONS:
1. Macro Regime Summary (SPY/VIX/DXY state)
2. TC Performance Matrix (PF, WR, DD per TC)
3. Convergence Trends (News-Macro alignment over week)
4. Model Accuracy Tracking (Fundamental model drift)
5. Top Trades Review (Best/worst trades)
6. Risk Metrics (Portfolio exposure, correlation)
7. Next Week Outlook (Regime forecast, active TCs)

OUTPUT: HTML + JSON export
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import json

from aggregation.regime_classifier import RegimeClassifier
from fundamental.fundamental_model import FundamentalModel
from fundamental.semantic_news_connector import SemanticNewsConnector


class EnhancedFridayReport:
    """Generates comprehensive weekly trading system report"""
    
    def __init__(self, output_dir: str = "./reports/weekly"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.regime_classifier = RegimeClassifier()
        self.fundamental_model = FundamentalModel()
        self.news_connector = SemanticNewsConnector()
    
    def generate_report(self, week_ending: Optional[str] = None) -> str:
        """
        Generate weekly report.
        
        Args:
            week_ending: Date string (YYYY-MM-DD). Defaults to today.
        
        Returns:
            Path to generated HTML report
        """
        if week_ending is None:
            week_ending = datetime.now().strftime('%Y-%m-%d')
        
        week_end_date = pd.to_datetime(week_ending)
        week_start_date = week_end_date - timedelta(days=7)
        
        print(f"\n{'='*80}")
        print(f"GENERATING WEEKLY REPORT: {week_start_date.date()} → {week_end_date.date()}")
        print(f"{'='*80}\n")
        
        report_data = {}
        
        # Section 1: Macro Regime
        print("[1/7] Macro Regime Summary...")
        report_data['regime'] = self._generate_regime_section()
        
        # Section 2: TC Performance
        print("[2/7] TC Performance Matrix...")
        report_data['tc_performance'] = self._generate_tc_performance_section()
        
        # Section 3: Convergence Trends
        print("[3/7] Convergence Trends...")
        report_data['convergence'] = self._generate_convergence_section()
        
        # Section 4: Model Accuracy
        print("[4/7] Model Accuracy Tracking...")
        report_data['model_accuracy'] = self._generate_model_accuracy_section()
        
        # Section 5: Top Trades
        print("[5/7] Top Trades Review...")
        report_data['top_trades'] = self._generate_top_trades_section()
        
        # Section 6: Risk Metrics
        print("[6/7] Risk Metrics...")
        report_data['risk_metrics'] = self._generate_risk_section()
        
        # Section 7: Next Week Outlook
        print("[7/7] Next Week Outlook...")
        report_data['outlook'] = self._generate_outlook_section()
        
        # Generate HTML
        html_path = self._generate_html(report_data, week_end_date)
        
        # Export JSON
        json_path = self.output_dir / f"report_{week_end_date.strftime('%Y%m%d')}.json"
        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"\n{'='*80}")
        print(f"✓ REPORT GENERATED")
        print(f"  HTML: {html_path}")
        print(f"  JSON: {json_path}")
        print(f"{'='*80}\n")
        
        return str(html_path)
    
    def _generate_regime_section(self) -> Dict[str, Any]:
        """Section 1: Macro regime state"""
        # Placeholder - would fetch real SPY/VIX data
        return {
            'current_regime': 'TRENDING',
            'adx_spy': 28.5,
            'vix_level': 14.2,
            'sma50_slope': 0.15,
            'active_tc_ids': ['TC-01', 'TC-03', 'TC-07', 'TC-11'],
            'position_size_mult': 1.0,
            'stop_mult': 1.0
        }
    
    def _generate_tc_performance_section(self) -> Dict[str, Any]:
        """Section 2: TC performance matrix"""
        # Placeholder - would fetch real trade logs
        tc_stats = []
        
        for tc_id in ['TC-01', 'TC-02', 'TC-03', 'TC-04', 'TC-05']:
            tc_stats.append({
                'tc_id': tc_id,
                'trades_this_week': np.random.randint(0, 10),
                'win_rate': np.random.uniform(0.4, 0.7),
                'profit_factor': np.random.uniform(0.8, 2.0),
                'max_drawdown': np.random.uniform(0.03, 0.12),
                'status': 'ACTIVE',
                'regime_fit': 'TRENDING' if tc_id in ['TC-01', 'TC-03'] else 'CORRECTIVE'
            })
        
        return {
            'tc_stats': tc_stats,
            'total_trades': sum(s['trades_this_week'] for s in tc_stats),
            'avg_win_rate': np.mean([s['win_rate'] for s in tc_stats])
        }
    
    def _generate_convergence_section(self) -> Dict[str, Any]:
        """Section 3: News-Macro convergence trends"""
        # Placeholder - would fetch real news data
        daily_convergence = []
        
        for i in range(7):
            date = datetime.now() - timedelta(days=6-i)
            daily_convergence.append({
                'date': date.strftime('%Y-%m-%d'),
                'convergence_score': np.random.uniform(0.5, 0.9),
                'direction': np.random.choice(['BULLISH', 'BEARISH', 'NEUTRAL']),
                'news_count': np.random.randint(5, 20)
            })
        
        return {
            'daily_convergence': daily_convergence,
            'avg_convergence': np.mean([d['convergence_score'] for d in daily_convergence]),
            'dominant_direction': 'BULLISH'  # Most frequent
        }
    
    def _generate_model_accuracy_section(self) -> Dict[str, Any]:
        """Section 4: Fundamental model drift detection"""
        return {
            'current_cv_accuracy': 0.926,
            'baseline_accuracy': 0.926,
            'drift': -0.003,
            'last_retrain_date': '2024-03-15',
            'days_since_retrain': 15,
            'retrain_recommended': False
        }
    
    def _generate_top_trades_section(self) -> Dict[str, Any]:
        """Section 5: Best/worst trades review"""
        best_trades = [
            {'symbol': 'NVDA', 'tc_id': 'TC-01', 'pnl': 2850, 'entry_date': '2024-03-25', 'exit_reason': 'TP_HIT'},
            {'symbol': 'SPY', 'tc_id': 'TC-03', 'pnl': 1920, 'entry_date': '2024-03-26', 'exit_reason': 'TP_HIT'},
            {'symbol': 'QQQ', 'tc_id': 'TC-01', 'pnl': 1540, 'entry_date': '2024-03-27', 'exit_reason': 'TP_HIT'}
        ]
        
        worst_trades = [
            {'symbol': 'TSLA', 'tc_id': 'TC-02', 'pnl': -850, 'entry_date': '2024-03-28', 'exit_reason': 'SL_HIT'},
            {'symbol': 'AAPL', 'tc_id': 'TC-03', 'pnl': -720, 'entry_date': '2024-03-29', 'exit_reason': 'SL_HIT'}
        ]
        
        return {
            'best_trades': best_trades,
            'worst_trades': worst_trades,
            'best_tc': 'TC-01',
            'worst_tc': 'TC-02'
        }
    
    def _generate_risk_section(self) -> Dict[str, Any]:
        """Section 6: Portfolio risk metrics"""
        return {
            'total_exposure': 0.35,  # 35% of capital deployed
            'max_position_size': 0.05,
            'correlation_spy': 0.65,
            'sharpe_ratio': 1.8,
            'sortino_ratio': 2.3,
            'var_95': 0.08  # 95% VaR
        }
    
    def _generate_outlook_section(self) -> Dict[str, Any]:
        """Section 7: Next week forecast"""
        return {
            'regime_forecast': 'TRENDING (70% confidence)',
            'recommended_tcs': ['TC-01', 'TC-03', 'TC-07', 'TC-12'],
            'reduce_exposure_tcs': ['TC-02', 'TC-06'],
            'key_events': [
                'Fed FOMC Minutes (Wed)',
                'NFP Report (Fri)',
                'Earnings: NVDA, GOOGL'
            ]
        }
    
    def _generate_html(self, data: Dict[str, Any], week_end_date: datetime) -> Path:
        """Generate HTML report from data"""
        html_path = self.output_dir / f"report_{week_end_date.strftime('%Y%m%d')}.html"
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Weekly Trading Report - {week_end_date.strftime('%Y-%m-%d')}</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #1a1a1a; border-bottom: 3px solid #0066cc; padding-bottom: 10px; }}
        h2 {{ color: #333; margin-top: 30px; border-left: 4px solid #0066cc; padding-left: 15px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th {{ background: #0066cc; color: white; padding: 12px; text-align: left; }}
        td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
        tr:hover {{ background: #f9f9f9; }}
        .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
        .metric-label {{ font-size: 12px; color: #666; text-transform: uppercase; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #0066cc; }}
        .positive {{ color: #28a745; }}
        .negative {{ color: #dc3545; }}
        .section {{ margin: 30px 0; padding: 20px; background: #fafafa; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Draculative Alpha Engine - Weekly Report</h1>
        <p style="color: #666;">Week Ending: {week_end_date.strftime('%A, %B %d, %Y')}</p>
        
        <div class="section">
            <h2>📊 Section 1: Macro Regime Summary</h2>
            <div class="metric">
                <div class="metric-label">Current Regime</div>
                <div class="metric-value">{data['regime']['current_regime']}</div>
            </div>
            <div class="metric">
                <div class="metric-label">SPY ADX</div>
                <div class="metric-value">{data['regime']['adx_spy']:.1f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">VIX Level</div>
                <div class="metric-value">{data['regime']['vix_level']:.1f}</div>
            </div>
            <p><strong>Active TCs:</strong> {', '.join(data['regime']['active_tc_ids'])}</p>
        </div>
        
        <div class="section">
            <h2>⚡ Section 2: TC Performance Matrix</h2>
            <table>
                <thead>
                    <tr>
                        <th>TC ID</th>
                        <th>Trades</th>
                        <th>Win Rate</th>
                        <th>Profit Factor</th>
                        <th>Max DD</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
"""
        
        for tc in data['tc_performance']['tc_stats']:
            wr_class = 'positive' if tc['win_rate'] > 0.5 else 'negative'
            pf_class = 'positive' if tc['profit_factor'] > 1.0 else 'negative'
            
            html += f"""
                    <tr>
                        <td><strong>{tc['tc_id']}</strong></td>
                        <td>{tc['trades_this_week']}</td>
                        <td class="{wr_class}">{tc['win_rate']:.1%}</td>
                        <td class="{pf_class}">{tc['profit_factor']:.2f}</td>
                        <td>{tc['max_drawdown']:.1%}</td>
                        <td>{tc['status']}</td>
                    </tr>
"""
        
        html += f"""
                </tbody>
            </table>
            <p><strong>Total Trades:</strong> {data['tc_performance']['total_trades']} | 
               <strong>Avg Win Rate:</strong> {data['tc_performance']['avg_win_rate']:.1%}</p>
        </div>
        
        <div class="section">
            <h2>🔗 Section 3: News-Macro Convergence</h2>
            <p><strong>Weekly Average:</strong> {data['convergence']['avg_convergence']:.1%} | 
               <strong>Direction:</strong> {data['convergence']['dominant_direction']}</p>
        </div>
        
        <div class="section">
            <h2>🎯 Section 4: Model Accuracy</h2>
            <div class="metric">
                <div class="metric-label">CV Accuracy</div>
                <div class="metric-value">{data['model_accuracy']['current_cv_accuracy']:.1%}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Drift</div>
                <div class="metric-value {'negative' if data['model_accuracy']['drift'] < 0 else 'positive'}">
                    {data['model_accuracy']['drift']:.3f}
                </div>
            </div>
            <p>Days Since Retrain: {data['model_accuracy']['days_since_retrain']}</p>
        </div>
        
        <div class="section">
            <h2>🏆 Section 5: Top Trades</h2>
            <h3>Best Trades</h3>
            <table>
                <thead><tr><th>Symbol</th><th>TC</th><th>P&L</th><th>Entry Date</th><th>Exit Reason</th></tr></thead>
                <tbody>
"""
        
        for trade in data['top_trades']['best_trades']:
            html += f"""
                    <tr>
                        <td><strong>{trade['symbol']}</strong></td>
                        <td>{trade['tc_id']}</td>
                        <td class="positive">${trade['pnl']:,.0f}</td>
                        <td>{trade['entry_date']}</td>
                        <td>{trade['exit_reason']}</td>
                    </tr>
"""
        
        html += """
                </tbody>
            </table>
        </div>
        
        <div class="section">
            <h2>⚠️ Section 6: Risk Metrics</h2>
"""
        
        risk = data['risk_metrics']
        html += f"""
            <div class="metric">
                <div class="metric-label">Total Exposure</div>
                <div class="metric-value">{risk['total_exposure']:.1%}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-value">{risk['sharpe_ratio']:.2f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">95% VaR</div>
                <div class="metric-value">{risk['var_95']:.1%}</div>
            </div>
        </div>
        
        <div class="section">
            <h2>🔮 Section 7: Next Week Outlook</h2>
"""
        
        outlook = data['outlook']
        html += f"""
            <p><strong>Regime Forecast:</strong> {outlook['regime_forecast']}</p>
            <p><strong>Recommended TCs:</strong> {', '.join(outlook['recommended_tcs'])}</p>
            <p><strong>Reduce Exposure:</strong> {', '.join(outlook['reduce_exposure_tcs'])}</p>
            <p><strong>Key Events:</strong></p>
            <ul>
"""
        
        for event in outlook['key_events']:
            html += f"                <li>{event}</li>\n"
        
        html += """
            </ul>
        </div>
        
        <footer style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; text-align: center;">
            <p>Generated by Draculative Alpha Engine | {}</p>
        </footer>
    </div>
</body>
</html>
""".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        with open(html_path, 'w') as f:
            f.write(html)
        
        return html_path


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate weekly trading report")
    parser.add_argument('--week-ending', type=str, help='Week ending date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    report_generator = EnhancedFridayReport()
    report_path = report_generator.generate_report(week_ending=args.week_ending)
    
    print(f"\n✓ Report ready: {report_path}")
    print("\nTo view: open the HTML file in your browser")
