"""
Semantic News Connector
========================

Replaces keyword-based news scoring with semantic similarity.

UPGRADE:
- Old: TF-IDF keyword matching → topic scores
- New: Sentence-transformers cosine similarity → convergence scores

INTEGRATION:
- Uses CorpusIngestor for PH Macro context retrieval
- Scores news items against transcript embeddings
- Returns alignment score (0-1) with evidence snippets
"""

import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    print("⚠️  Missing dependency. Install: pip install sentence-transformers")
    exit(1)

from fundamental.corpus_ingestor import CorpusIngestor


class SemanticNewsConnector:
    """Semantic news-to-macro alignment scoring"""
    
    def __init__(self, corpus_db_path: str = "./data/chroma_db"):
        """
        Initialize semantic news connector.
        
        Args:
            corpus_db_path: Path to ChromaDB with PH Macro corpus
        """
        print("Loading sentence-transformers model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("Connecting to PH Macro corpus...")
        self.corpus = CorpusIngestor(db_path=corpus_db_path)
        
        print("✓ Semantic connector ready")
    
    def score_news_item(self, news_text: str, n_context: int = 5) -> Dict[str, Any]:
        """
        Score news item against PH Macro corpus.
        
        Args:
            news_text: News headline + summary
            n_context: Number of corpus chunks to retrieve
        
        Returns:
            {
                'alignment_score': float (0-1),
                'confidence': float (0-1),
                'direction': str (BULLISH/BEARISH/NEUTRAL),
                'evidence': list of matching transcript snippets
            }
        """
        # Retrieve semantically similar transcript chunks
        context_results = self.corpus.search(news_text, n_results=n_context)
        
        if len(context_results) == 0:
            return {
                'alignment_score': 0.5,
                'confidence': 0.0,
                'direction': 'NEUTRAL',
                'evidence': []
            }
        
        # Calculate average similarity (distance is L2, convert to similarity)
        distances = [r['distance'] for r in context_results]
        avg_distance = np.mean(distances)
        
        # Convert distance to similarity (0-1 scale)
        # Typical distance range: 0.3 (very similar) to 1.5 (dissimilar)
        similarity = max(0, 1 - (avg_distance / 1.5))
        
        # Determine direction from top match
        top_match = context_results[0]['text'].lower()
        
        bullish_keywords = ['bullish', 'buy', 'long', 'rally', 'upside', 'positive', 'growth', 'strong']
        bearish_keywords = ['bearish', 'sell', 'short', 'decline', 'downside', 'negative', 'weak', 'risk']
        
        bullish_count = sum(1 for kw in bullish_keywords if kw in top_match)
        bearish_count = sum(1 for kw in bearish_keywords if kw in top_match)
        
        if bullish_count > bearish_count:
            direction = 'BULLISH'
        elif bearish_count > bullish_count:
            direction = 'BEARISH'
        else:
            direction = 'NEUTRAL'
        
        # Evidence snippets
        evidence = [
            {
                'text': r['text'][:200] + '...',
                'video': r['metadata']['title'],
                'date': r['metadata']['upload_date'],
                'similarity': 1 - (r['distance'] / 1.5)
            }
            for r in context_results[:3]
        ]
        
        return {
            'alignment_score': similarity,
            'confidence': similarity,  # High similarity = high confidence
            'direction': direction,
            'evidence': evidence
        }
    
    def score_news_batch(self, news_items: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Score multiple news items.
        
        Args:
            news_items: List of {headline, summary, published_date}
        
        Returns:
            List of scored results
        """
        results = []
        
        for item in news_items:
            text = f"{item.get('headline', '')} {item.get('summary', '')}"
            score = self.score_news_item(text)
            
            results.append({
                **item,
                **score
            })
        
        return results
    
    def get_convergence_signal(self, news_items: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Aggregate news batch into single convergence signal.
        
        Args:
            news_items: List of news items to aggregate
        
        Returns:
            {
                'convergence_score': float (0-1),
                'direction': str,
                'conviction': str (LOW/MODERATE/HIGH),
                'supporting_count': int,
                'conflicting_count': int
            }
        """
        if len(news_items) == 0:
            return {
                'convergence_score': 0.5,
                'direction': 'NEUTRAL',
                'conviction': 'NONE',
                'supporting_count': 0,
                'conflicting_count': 0
            }
        
        scored_items = self.score_news_batch(news_items)
        
        # Aggregate scores
        bullish_scores = [s['alignment_score'] for s in scored_items if s['direction'] == 'BULLISH']
        bearish_scores = [s['alignment_score'] for s in scored_items if s['direction'] == 'BEARISH']
        
        bullish_avg = np.mean(bullish_scores) if bullish_scores else 0
        bearish_avg = np.mean(bearish_scores) if bearish_scores else 0
        
        # Determine overall direction
        if bullish_avg > bearish_avg:
            direction = 'BULLISH'
            convergence_score = bullish_avg
            supporting_count = len(bullish_scores)
            conflicting_count = len(bearish_scores)
        else:
            direction = 'BEARISH'
            convergence_score = bearish_avg
            supporting_count = len(bearish_scores)
            conflicting_count = len(bullish_scores)
        
        # Conviction based on score + agreement
        agreement_ratio = supporting_count / len(scored_items)
        
        if convergence_score > 0.7 and agreement_ratio > 0.7:
            conviction = 'HIGH'
        elif convergence_score > 0.5 and agreement_ratio > 0.5:
            conviction = 'MODERATE'
        else:
            conviction = 'LOW'
        
        return {
            'convergence_score': convergence_score,
            'direction': direction,
            'conviction': conviction,
            'supporting_count': supporting_count,
            'conflicting_count': conflicting_count
        }


# ══════════════════════════════════════════════════════════════════════════════
# CLI TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n🧪 Semantic News Connector — Test Mode\n")
    
    connector = SemanticNewsConnector()
    
    # Test news items
    test_news = [
        {
            'headline': 'Fed signals rate cuts amid cooling inflation',
            'summary': 'Federal Reserve Chair Jerome Powell indicated potential rate cuts in 2024 as inflation shows signs of cooling.',
            'published_date': '2024-03-15'
        },
        {
            'headline': 'China economic data disappoints, raising stimulus speculation',
            'summary': 'Chinas manufacturing PMI fell below expectations, increasing calls for government stimulus.',
            'published_date': '2024-03-14'
        },
        {
            'headline': 'US dollar strengthens on strong jobs report',
            'summary': 'Non-farm payrolls beat estimates, boosting dollar against major currencies.',
            'published_date': '2024-03-13'
        }
    ]
    
    print("Scoring news items...\n")
    
    for item in test_news:
        print(f"Headline: {item['headline']}")
        score = connector.score_news_item(f"{item['headline']} {item['summary']}")
        
        print(f"  Direction: {score['direction']}")
        print(f"  Alignment: {score['alignment_score']:.2%}")
        print(f"  Confidence: {score['confidence']:.2%}")
        print(f"  Evidence:")
        for ev in score['evidence'][:2]:
            print(f"    • {ev['video']} ({ev['similarity']:.2%} match)")
        print()
    
    print("\nConvergence Signal:")
    convergence = connector.get_convergence_signal(test_news)
    print(f"  Direction: {convergence['direction']}")
    print(f"  Score: {convergence['convergence_score']:.2%}")
    print(f"  Conviction: {convergence['conviction']}")
    print(f"  Supporting: {convergence['supporting_count']}, Conflicting: {convergence['conflicting_count']}")
