#!/usr/bin/env python3
"""
News & Events Tracker

Scrapes and analyzes global news, correlating historical events with market impact.
Uses multiple news sources and AI-based sentiment analysis.

Event Categories:
- Fed/Central Bank decisions
- Earnings announcements
- Geopolitical events (wars, sanctions)
- Economic data releases
- Sector-specific news
- Company-specific news
"""

import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False


# =============================================================================
# HISTORICAL EVENT DATABASE
# =============================================================================

# Major historical events and their market impact
HISTORICAL_EVENTS = {
    # Fed Rate Decisions
    "fed_rate_hike": {
        "examples": [
            {"date": "2022-03-16", "description": "Fed raises rates 25bp", "spy_1d": -0.7, "spy_5d": 2.5, "qqq_1d": -1.1, "qqq_5d": 3.2},
            {"date": "2022-05-04", "description": "Fed raises rates 50bp", "spy_1d": 3.0, "spy_5d": -5.2, "qqq_1d": 3.2, "qqq_5d": -8.1},
            {"date": "2022-06-15", "description": "Fed raises rates 75bp", "spy_1d": -3.3, "spy_5d": 6.1, "qqq_1d": -4.1, "qqq_5d": 7.2},
            {"date": "2022-09-21", "description": "Fed raises rates 75bp", "spy_1d": -1.7, "spy_5d": -2.8, "qqq_1d": -1.8, "qqq_5d": -3.1},
            {"date": "2023-07-26", "description": "Fed raises rates 25bp", "spy_1d": 0.0, "spy_5d": -0.5, "qqq_1d": -0.1, "qqq_5d": -1.2},
        ],
        "avg_spy_1d": -0.5,
        "avg_spy_5d": 0.0,
        "typical_sectors_hit": ["XLF", "XLRE", "XLU"],
        "typical_sectors_benefit": ["XLE"],
        "description": "Fed rate hikes typically cause initial volatility. Tech and rate-sensitive sectors often decline."
    },
    
    "fed_rate_cut": {
        "examples": [
            {"date": "2020-03-03", "description": "Emergency 50bp cut (COVID)", "spy_1d": -2.8, "spy_5d": -8.2, "qqq_1d": -2.1, "qqq_5d": -8.5},
            {"date": "2020-03-15", "description": "Emergency 100bp cut (COVID)", "spy_1d": -12.0, "spy_5d": -5.0, "qqq_1d": -12.3, "qqq_5d": -2.1},
            {"date": "2019-07-31", "description": "Fed cuts rates 25bp", "spy_1d": -1.1, "spy_5d": -3.1, "qqq_1d": -1.4, "qqq_5d": -3.8},
            {"date": "2024-09-18", "description": "Fed cuts rates 50bp", "spy_1d": 1.7, "spy_5d": 1.2, "qqq_1d": 2.5, "qqq_5d": 1.8},
        ],
        "avg_spy_1d": -3.5,
        "avg_spy_5d": -3.8,
        "typical_sectors_benefit": ["XLK", "XLRE", "XLY"],
        "description": "Emergency cuts signal crisis (bearish). Planned cuts in healthy economy are bullish."
    },
    
    # Geopolitical Events
    "war_outbreak": {
        "examples": [
            {"date": "2022-02-24", "description": "Russia invades Ukraine", "spy_1d": 1.5, "spy_5d": -1.5, "oil_1d": 8.0, "gold_1d": 2.5},
            {"date": "2023-10-07", "description": "Hamas attacks Israel", "spy_1d": 0.5, "spy_5d": -0.8, "oil_1d": 4.2, "gold_1d": 1.8},
        ],
        "avg_spy_1d": 1.0,
        "typical_sectors_benefit": ["XLE", "XLI", "GLD"],
        "typical_sectors_hit": ["XLY", "XLC"],
        "description": "Wars cause flight to safety (gold, defense), oil spikes. Initial shock often bought."
    },
    
    "tariff_announcement": {
        "examples": [
            {"date": "2018-03-22", "description": "Trump tariffs on China", "spy_1d": -2.5, "spy_5d": -3.8},
            {"date": "2019-05-05", "description": "Trump raises China tariffs to 25%", "spy_1d": -2.4, "spy_5d": -2.1},
            {"date": "2025-02-01", "description": "Trump tariffs on Mexico/Canada", "spy_1d": -0.5, "spy_5d": -1.2},
        ],
        "avg_spy_1d": -1.8,
        "typical_sectors_hit": ["XLI", "XLY", "XLK"],
        "description": "Tariffs create uncertainty, hurt importers and exporters."
    },
    
    # Economic Data
    "cpi_hot": {
        "examples": [
            {"date": "2022-06-10", "description": "CPI 8.6% YoY (surprise)", "spy_1d": -2.9, "spy_5d": -6.8},
            {"date": "2022-09-13", "description": "CPI 8.3% (higher than expected)", "spy_1d": -4.3, "spy_5d": -2.1},
        ],
        "avg_spy_1d": -3.6,
        "description": "Hot inflation = more Fed tightening expected. Very bearish."
    },
    
    "cpi_cool": {
        "examples": [
            {"date": "2022-11-10", "description": "CPI 7.7% (cooler than expected)", "spy_1d": 5.5, "spy_5d": 4.2},
            {"date": "2023-11-14", "description": "CPI 3.2% (below expectations)", "spy_1d": 1.9, "spy_5d": 2.8},
        ],
        "avg_spy_1d": 3.7,
        "description": "Cool inflation = Fed can ease. Very bullish."
    },
    
    "jobs_strong": {
        "examples": [
            {"date": "2023-01-06", "description": "NFP 517K (much higher than expected)", "spy_1d": -1.1, "spy_5d": -0.3},
            {"date": "2024-01-05", "description": "NFP 216K (above expectations)", "spy_1d": 0.2, "spy_5d": 1.5},
        ],
        "avg_spy_1d": -0.4,
        "description": "Strong jobs = good economy but more Fed tightening risk."
    },
    
    "jobs_weak": {
        "examples": [
            {"date": "2024-08-02", "description": "NFP 114K (much below expected)", "spy_1d": -1.8, "spy_5d": 0.5},
        ],
        "avg_spy_1d": -1.8,
        "description": "Weak jobs = recession fears but Fed dovish. Mixed impact."
    },
    
    # Tech/AI Events
    "ai_breakthrough": {
        "examples": [
            {"date": "2022-11-30", "description": "ChatGPT launch", "msft_5d": 2.1, "nvda_5d": 5.2, "googl_5d": -3.1},
            {"date": "2023-01-23", "description": "Microsoft $10B OpenAI investment", "msft_1d": 0.5, "nvda_1d": 4.1},
            {"date": "2024-01-08", "description": "Nvidia CES AI announcements", "nvda_1d": 6.4},
        ],
        "typical_stocks_benefit": ["NVDA", "MSFT", "AMD", "GOOGL", "META"],
        "description": "AI breakthroughs massively benefit chip makers and big tech."
    },
    
    # Sector Specific
    "bank_failure": {
        "examples": [
            {"date": "2023-03-10", "description": "Silicon Valley Bank collapse", "spy_1d": -1.5, "xlf_1d": -4.1, "xlf_5d": -8.2},
            {"date": "2023-03-12", "description": "Signature Bank closure", "spy_1d": -0.2, "xlf_1d": -2.1},
        ],
        "avg_xlf_1d": -3.1,
        "typical_sectors_hit": ["XLF"],
        "typical_sectors_benefit": ["TLT"],
        "description": "Bank failures cause flight to safety, crush financial sector."
    },
    
    # Earnings Related
    "earnings_beat_mega_cap": {
        "examples": [
            {"date": "2024-02-22", "description": "NVDA earnings blowout", "nvda_1d": 16.4, "spy_1d": 2.1},
            {"date": "2024-04-25", "description": "META earnings beat", "meta_1d": 14.2},
        ],
        "description": "Mega cap earnings beats can lift entire market."
    },
    
    "earnings_miss_mega_cap": {
        "examples": [
            {"date": "2022-07-27", "description": "META misses, metaverse concerns", "meta_1d": -26.4, "spy_1d": 1.2},
            {"date": "2024-04-24", "description": "META capex concerns", "meta_1d": -10.6},
        ],
        "description": "Mega cap misses can create sector-wide fear."
    },
}


# Event category keywords for matching
EVENT_KEYWORDS = {
    "fed_rate_hike": ["fed", "rate hike", "hawkish", "tightening", "basis points", "fomc", "powell", "rate increase"],
    "fed_rate_cut": ["fed", "rate cut", "dovish", "easing", "rate reduction", "fomc", "powell"],
    "war_outbreak": ["war", "invasion", "attack", "military", "conflict", "troops", "missile", "strike"],
    "tariff_announcement": ["tariff", "trade war", "import tax", "customs", "duties", "trade restrictions"],
    "cpi_hot": ["cpi", "inflation", "higher than expected", "inflation surge", "price increase"],
    "cpi_cool": ["cpi", "inflation", "lower than expected", "inflation cooling", "disinflation"],
    "jobs_strong": ["jobs report", "nonfarm payrolls", "employment", "hiring", "labor market strong"],
    "jobs_weak": ["jobs report", "unemployment", "layoffs", "hiring freeze", "job cuts"],
    "ai_breakthrough": ["artificial intelligence", "ai", "chatgpt", "gpt", "llm", "machine learning", "nvidia"],
    "bank_failure": ["bank failure", "bank collapse", "bank run", "fdic", "bailout"],
    "earnings_beat_mega_cap": ["earnings beat", "revenue beat", "guidance raise", "blowout quarter"],
    "earnings_miss_mega_cap": ["earnings miss", "revenue miss", "guidance cut", "disappointing quarter"],
}


@dataclass
class NewsItem:
    """Single news item."""
    title: str
    source: str
    url: str
    published: str
    summary: str
    sentiment: float  # -1 to 1
    relevance: float  # 0 to 1
    tickers: List[str]
    event_type: Optional[str]
    predicted_impact: Optional[Dict]


@dataclass
class NewsAnalysis:
    """Complete news analysis for a ticker."""
    ticker: str
    timestamp: str
    overall_sentiment: float
    news_count: int
    event_signals: List[Dict]
    top_news: List[NewsItem]
    sector_news: List[NewsItem]
    macro_news: List[NewsItem]
    predicted_impact: Dict


class NewsTracker:
    """Track and analyze news for market impact prediction."""
    
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.news_items = []
        
    def fetch_news(self) -> List[Dict]:
        """Fetch news from multiple sources."""
        all_news = []
        
        # Yahoo Finance News
        yahoo_news = self._fetch_yahoo_news()
        all_news.extend(yahoo_news)
        
        # RSS Feeds (if available)
        rss_news = self._fetch_rss_feeds()
        all_news.extend(rss_news)
        
        return all_news
    
    def _fetch_yahoo_news(self) -> List[Dict]:
        """Fetch news from Yahoo Finance."""
        try:
            import yfinance as yf
            stock = yf.Ticker(self.ticker)
            news = stock.news
            
            result = []
            for item in news[:20]:
                result.append({
                    'title': item.get('title', ''),
                    'source': item.get('publisher', 'Yahoo Finance'),
                    'url': item.get('link', ''),
                    'published': datetime.fromtimestamp(item.get('providerPublishTime', 0)).isoformat(),
                    'summary': item.get('title', ''),  # Yahoo doesn't always provide summary
                    'type': item.get('type', 'STORY')
                })
            return result
        except Exception as e:
            print(f"  Yahoo news error: {e}")
            return []
    
    def _fetch_rss_feeds(self) -> List[Dict]:
        """Fetch news from financial RSS feeds."""
        feeds = [
            f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={self.ticker}&region=US&lang=en-US",
        ]
        
        all_items = []
        
        if not HAS_REQUESTS or not HAS_BS4:
            return all_items
            
        for feed_url in feeds:
            try:
                response = requests.get(feed_url, timeout=10)
                soup = BeautifulSoup(response.content, 'xml')
                
                for item in soup.find_all('item')[:10]:
                    all_items.append({
                        'title': item.title.text if item.title else '',
                        'source': 'RSS',
                        'url': item.link.text if item.link else '',
                        'published': item.pubDate.text if item.pubDate else '',
                        'summary': item.description.text if item.description else '',
                        'type': 'RSS'
                    })
            except Exception as e:
                continue
        
        return all_items
    
    def analyze_sentiment(self, text: str) -> float:
        """
        Simple sentiment analysis using keyword matching.
        Returns score from -1 (bearish) to 1 (bullish).
        """
        text_lower = text.lower()
        
        bullish_words = [
            'surge', 'soar', 'jump', 'rally', 'gain', 'rise', 'up', 'higher', 'bull',
            'beat', 'exceed', 'strong', 'growth', 'profit', 'upgrade', 'buy',
            'outperform', 'positive', 'optimistic', 'boom', 'record high', 'breakthrough'
        ]
        
        bearish_words = [
            'fall', 'drop', 'plunge', 'crash', 'decline', 'down', 'lower', 'bear',
            'miss', 'weak', 'loss', 'downgrade', 'sell', 'underperform', 'negative',
            'pessimistic', 'recession', 'fear', 'concern', 'warning', 'risk', 'cut'
        ]
        
        neutral_modifiers = ['may', 'could', 'might', 'potential', 'uncertain']
        
        bullish_count = sum(1 for word in bullish_words if word in text_lower)
        bearish_count = sum(1 for word in bearish_words if word in text_lower)
        neutral_count = sum(1 for word in neutral_modifiers if word in text_lower)
        
        # Weight down if neutral modifiers present
        modifier = 0.7 if neutral_count > 0 else 1.0
        
        total = bullish_count + bearish_count
        if total == 0:
            return 0.0
        
        sentiment = (bullish_count - bearish_count) / total * modifier
        return round(max(-1, min(1, sentiment)), 2)
    
    def classify_event(self, text: str) -> Optional[Tuple[str, float]]:
        """
        Classify news into event category.
        Returns (event_type, confidence) or None.
        """
        text_lower = text.lower()
        
        best_match = None
        best_score = 0
        
        for event_type, keywords in EVENT_KEYWORDS.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            if matches > best_score:
                best_score = matches
                best_match = event_type
        
        if best_match and best_score >= 2:
            confidence = min(1.0, best_score / 4)
            return (best_match, confidence)
        
        return None
    
    def predict_impact(self, event_type: str) -> Dict:
        """Predict market impact based on historical events."""
        if event_type not in HISTORICAL_EVENTS:
            return {"prediction": "unknown", "confidence": 0}
        
        event_data = HISTORICAL_EVENTS[event_type]
        
        return {
            "event_type": event_type,
            "description": event_data.get("description", ""),
            "avg_spy_1d": event_data.get("avg_spy_1d", 0),
            "avg_spy_5d": event_data.get("avg_spy_5d", 0),
            "sectors_benefit": event_data.get("typical_sectors_benefit", []),
            "sectors_hurt": event_data.get("typical_sectors_hit", []),
            "historical_examples": event_data.get("examples", [])[:3],
            "confidence": 0.7
        }
    
    def extract_tickers(self, text: str) -> List[str]:
        """Extract stock tickers mentioned in text."""
        # Common ticker patterns
        pattern = r'\b[A-Z]{1,5}\b'
        potential_tickers = re.findall(pattern, text)
        
        # Filter common words that aren't tickers
        common_words = {'A', 'I', 'CEO', 'CFO', 'IPO', 'ETF', 'US', 'UK', 'EU', 'AI', 'IT', 
                       'THE', 'AND', 'FOR', 'NEW', 'SEC', 'FED', 'GDP', 'CPI', 'NFP'}
        
        tickers = [t for t in potential_tickers if t not in common_words and len(t) >= 2]
        
        # Always include the primary ticker
        if self.ticker not in tickers:
            tickers.insert(0, self.ticker)
        
        return tickers[:5]
    
    def analyze(self) -> NewsAnalysis:
        """Complete news analysis."""
        print(f"Analyzing news for {self.ticker}...")
        
        # Fetch news
        raw_news = self.fetch_news()
        print(f"  Found {len(raw_news)} news items")
        
        # Process each news item
        processed_news = []
        event_signals = []
        
        for item in raw_news:
            title = item.get('title', '')
            summary = item.get('summary', title)
            full_text = f"{title} {summary}"
            
            # Sentiment
            sentiment = self.analyze_sentiment(full_text)
            
            # Event classification
            event_class = self.classify_event(full_text)
            event_type = event_class[0] if event_class else None
            
            # Predicted impact
            predicted_impact = self.predict_impact(event_type) if event_type else None
            
            if event_type:
                event_signals.append({
                    "event_type": event_type,
                    "confidence": event_class[1],
                    "headline": title[:100],
                    "predicted_impact": predicted_impact
                })
            
            # Relevance (higher if ticker mentioned or sector related)
            tickers = self.extract_tickers(full_text)
            relevance = 1.0 if self.ticker in tickers else 0.5
            
            news_item = NewsItem(
                title=title,
                source=item.get('source', 'Unknown'),
                url=item.get('url', ''),
                published=item.get('published', ''),
                summary=summary[:200],
                sentiment=sentiment,
                relevance=relevance,
                tickers=tickers,
                event_type=event_type,
                predicted_impact=predicted_impact
            )
            processed_news.append(news_item)
        
        # Calculate overall sentiment
        if processed_news:
            overall_sentiment = np.mean([n.sentiment * n.relevance for n in processed_news])
        else:
            overall_sentiment = 0.0
        
        # Categorize news
        top_news = sorted(processed_news, key=lambda x: x.relevance, reverse=True)[:5]
        sector_news = [n for n in processed_news if n.event_type in ['ai_breakthrough', 'earnings_beat_mega_cap', 'earnings_miss_mega_cap']][:3]
        macro_news = [n for n in processed_news if n.event_type in ['fed_rate_hike', 'fed_rate_cut', 'cpi_hot', 'cpi_cool', 'war_outbreak']][:3]
        
        # Aggregate predicted impact
        if event_signals:
            best_event = max(event_signals, key=lambda x: x['confidence'])
            predicted_impact = best_event.get('predicted_impact', {})
        else:
            predicted_impact = {"prediction": "neutral", "confidence": 0.3}
        
        return NewsAnalysis(
            ticker=self.ticker,
            timestamp=datetime.now().isoformat(),
            overall_sentiment=round(overall_sentiment, 2),
            news_count=len(processed_news),
            event_signals=event_signals,
            top_news=top_news,
            sector_news=sector_news,
            macro_news=macro_news,
            predicted_impact=predicted_impact
        )
    
    def to_dict(self, analysis: NewsAnalysis) -> Dict:
        """Convert to dictionary."""
        return {
            'ticker': analysis.ticker,
            'timestamp': analysis.timestamp,
            'overall_sentiment': analysis.overall_sentiment,
            'news_count': analysis.news_count,
            'event_signals': analysis.event_signals,
            'top_news': [asdict(n) for n in analysis.top_news],
            'macro_news': [asdict(n) for n in analysis.macro_news],
            'predicted_impact': analysis.predicted_impact
        }


def analyze_news(ticker: str) -> NewsAnalysis:
    """Convenience function to analyze news for a ticker."""
    tracker = NewsTracker(ticker)
    return tracker.analyze()


if __name__ == '__main__':
    import sys
    
    ticker = sys.argv[1] if len(sys.argv) > 1 else "IONQ"
    
    tracker = NewsTracker(ticker)
    analysis = tracker.analyze()
    
    print("\n" + "="*60)
    print(f"NEWS ANALYSIS: {ticker}")
    print("="*60)
    
    print(f"\n📰 OVERALL SENTIMENT: {analysis.overall_sentiment:+.2f}")
    print(f"   News Items Analyzed: {analysis.news_count}")
    
    if analysis.event_signals:
        print(f"\n⚡ EVENT SIGNALS DETECTED:")
        for event in analysis.event_signals[:3]:
            print(f"   - {event['event_type']} (confidence: {event['confidence']:.1%})")
            print(f"     {event['headline'][:80]}...")
            if event.get('predicted_impact'):
                impact = event['predicted_impact']
                print(f"     Historical SPY impact: {impact.get('avg_spy_1d', 'N/A')}% (1D)")
    
    print(f"\n📊 TOP NEWS:")
    for news in analysis.top_news[:3]:
        sentiment_emoji = "🟢" if news.sentiment > 0.2 else "🔴" if news.sentiment < -0.2 else "🟡"
        print(f"   {sentiment_emoji} [{news.sentiment:+.2f}] {news.title[:70]}...")
        print(f"      Source: {news.source}")
    
    if analysis.predicted_impact and analysis.predicted_impact.get('event_type'):
        print(f"\n🎯 PREDICTED MARKET IMPACT:")
        impact = analysis.predicted_impact
        print(f"   Event Type: {impact.get('event_type')}")
        print(f"   Description: {impact.get('description', 'N/A')}")
        print(f"   Expected SPY Move (1D): {impact.get('avg_spy_1d', 'N/A')}%")
        if impact.get('sectors_benefit'):
            print(f"   Sectors to Benefit: {', '.join(impact['sectors_benefit'])}")
        if impact.get('sectors_hurt'):
            print(f"   Sectors at Risk: {', '.join(impact['sectors_hurt'])}")
    
    print("\n" + "="*60)
