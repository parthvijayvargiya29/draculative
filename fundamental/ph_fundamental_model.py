"""
fundamental/ph_fundamental_model.py
======================================
Directional ML model for the US equity market, trained exclusively on
PH macro video transcripts.

Architecture
------------
1.  Corpus chunking
    Each transcript is split into ~150-word chunks.  Each chunk is labelled
    with the dominant direction from the TranscriptTheme for that file
    (BULLISH → +1, BEARISH → -1, NEUTRAL → 0).

2.  Feature extraction
    TF-IDF (max 3000 features, unigrams + bigrams).

3.  Classifier
    Multi-class LogisticRegression (3 classes: BULLISH / NEUTRAL / BEARISH).
    Calibrated with CalibratedClassifierCV (isotonic) to produce
    meaningful probabilities.

4.  Prediction
    Given a free-text news snippet or macro summary, returns:
      DirectionalPrediction(direction, confidence, topic_breakdown)

Model is cached as data/ph_fundamental_model.pkl.
Re-train with refresh=True or when new transcripts are added.

Usage
-----
    from fundamental.ph_fundamental_model import PHFundamentalModel
    model = PHFundamentalModel()
    model.train()
    pred = model.predict("Fed signals rate cuts are off the table for 2025.")
    print(pred.direction, pred.confidence)
"""

from __future__ import annotations

import json
import pickle
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_ROOT      = Path(__file__).parent.parent
MODEL_PATH = _ROOT / "data" / "ph_fundamental_model.pkl"
META_PATH  = _ROOT / "data" / "ph_fundamental_model_meta.json"

sys.path.insert(0, str(_ROOT))
from fundamental.ph_transcript_parser import (
    TranscriptCorpus,
    load_corpus,
    TOPIC_SEEDS,
    BULLISH_WORDS,
    BEARISH_WORDS,
    _direction,
)

CHUNK_WORDS    = 150      # approximate words per training chunk
MIN_CHUNKS     = 5        # minimum chunks needed to train
LABEL_MAP      = {"BULLISH": 1, "NEUTRAL": 0, "BEARISH": -1}
LABEL_INV      = {v: k for k, v in LABEL_MAP.items()}


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class DirectionalPrediction:
    direction:        str          # BULLISH | NEUTRAL | BEARISH
    confidence:       float        # 0.0 – 1.0
    bull_prob:        float
    neutral_prob:     float
    bear_prob:        float
    topic_breakdown:  Dict[str, str] = field(default_factory=dict)
    input_text_preview: str = ""


@dataclass
class ModelMeta:
    trained_at:    str
    n_chunks:      int
    n_transcripts: int
    accuracy:      float
    classes:       List[str]
    vocab_size:    int


# ── Chunking ──────────────────────────────────────────────────────────────────

def _chunk_text(text: str, chunk_words: int = CHUNK_WORDS) -> List[str]:
    words = text.split()
    return [" ".join(words[i: i + chunk_words]) for i in range(0, len(words), chunk_words)
            if len(words[i: i + chunk_words]) >= 30]


def _rule_direction(text: str) -> int:
    """Fast rule-based direction for a chunk — used to cross-validate labels."""
    s = text.lower()
    bull = sum(1 for w in BULLISH_WORDS if w in s)
    bear = sum(1 for w in BEARISH_WORDS if w in s)
    diff = bull - bear
    if diff > 1:
        return 1
    if diff < -1:
        return -1
    return 0


def build_training_data(
    corpus: TranscriptCorpus,
) -> Tuple[List[str], List[int]]:
    """
    Build (X, y) from the corpus.
    Each chunk is labelled by the majority vote of:
      a) file-level overall_direction (from the parser)
      b) chunk-level rule-based score
    This gives more granular labels than using only file-level direction.
    """
    X: List[str] = []
    y: List[int] = []

    transcript_dir = _ROOT / "transcriptions" / "PH_macro" / "transcripts"

    for theme in corpus.files:
        txt_path = transcript_dir / theme.filename
        if not txt_path.exists():
            continue
        text = txt_path.read_text(encoding="utf-8", errors="replace")
        file_label = LABEL_MAP.get(theme.overall_direction, 0)

        for chunk in _chunk_text(text):
            rule_label  = _rule_direction(chunk)
            # Blend: if rule-based agrees → confident; if neutral → fall back to file label
            if rule_label == 0:
                label = file_label
            else:
                label = rule_label
            X.append(chunk)
            y.append(label)

    return X, y


# ── Model ─────────────────────────────────────────────────────────────────────

class PHFundamentalModel:
    """
    Directional prediction model trained on PH macro transcripts.
    Falls back gracefully to rule-based prediction when sklearn is unavailable
    or not enough training data exists.
    """

    def __init__(self):
        self._vectorizer  = None
        self._classifier  = None
        self._meta: Optional[ModelMeta] = None
        self._trained     = False
        self._corpus: Optional[TranscriptCorpus] = None

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, refresh: bool = False) -> ModelMeta:
        """
        Train the model.  Loads corpus, builds chunks, fits TF-IDF +
        LogisticRegression.  Saves to MODEL_PATH.  Returns ModelMeta.
        """
        # Load from disk if already trained and not refreshing
        if not refresh and MODEL_PATH.exists() and META_PATH.exists():
            try:
                self._load()
                return self._meta  # type: ignore[return-value]
            except Exception:
                pass

        # Build corpus
        corpus = load_corpus(refresh=refresh)
        self._corpus = corpus

        if corpus.source_count == 0:
            print("  [WARN] No transcripts found. Model will use rule-based fallback.")
            self._trained = False
            return ModelMeta(
                trained_at="",
                n_chunks=0,
                n_transcripts=0,
                accuracy=0.0,
                classes=[],
                vocab_size=0,
            )

        X, y = build_training_data(corpus)

        if len(X) < MIN_CHUNKS:
            print(f"  [WARN] Only {len(X)} training chunks — rule-based fallback.")
            self._trained = False
            return ModelMeta(
                trained_at="",
                n_chunks=len(X),
                n_transcripts=corpus.source_count,
                accuracy=0.0,
                classes=[],
                vocab_size=0,
            )

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.model_selection import cross_val_score
            from sklearn.preprocessing import LabelEncoder
            import datetime

            self._vectorizer = TfidfVectorizer(
                max_features=3000,
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=2,
            )
            X_mat = self._vectorizer.fit_transform(X)
            y_arr = np.array(y)

            base = LogisticRegression(
                C=1.0,
                max_iter=500,
                class_weight="balanced",
                solver="lbfgs",
                multi_class="multinomial",
            )
            # Calibrate for proper probabilities
            self._classifier = CalibratedClassifierCV(base, cv=min(3, len(set(y))), method="isotonic")
            self._classifier.fit(X_mat, y_arr)

            # Cross-val accuracy estimate
            try:
                cv_scores = cross_val_score(
                    self._classifier, X_mat, y_arr, cv=min(3, len(X) // 10 + 1), scoring="accuracy"
                )
                accuracy = float(np.mean(cv_scores))
            except Exception:
                accuracy = 0.0

            vocab_size = len(self._vectorizer.vocabulary_)
            classes    = [LABEL_INV.get(int(c), str(c)) for c in self._classifier.classes_]

            self._meta = ModelMeta(
                trained_at=datetime.datetime.now().isoformat(),
                n_chunks=len(X),
                n_transcripts=corpus.source_count,
                accuracy=round(accuracy, 4),
                classes=classes,
                vocab_size=vocab_size,
            )
            self._trained = True
            self._save()

            print(f"  ✅ Model trained: {len(X)} chunks, {corpus.source_count} transcripts, "
                  f"CV accuracy={accuracy:.2%}")
            return self._meta

        except ImportError:
            print("  [WARN] scikit-learn not installed — rule-based fallback active.")
            self._trained = False
            return ModelMeta(
                trained_at="",
                n_chunks=len(X),
                n_transcripts=corpus.source_count,
                accuracy=0.0,
                classes=[],
                vocab_size=0,
            )

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, text: str) -> DirectionalPrediction:
        """
        Predict directional bias for a free-text news item or macro summary.
        Falls back to rule-based scoring if model not trained.
        """
        preview = text[:120].replace("\n", " ") + ("…" if len(text) > 120 else "")

        # Topic breakdown (always rule-based)
        topic_breakdown: Dict[str, str] = {}
        for topic, seeds in TOPIC_SEEDS.items():
            s = text.lower()
            if any(seed in s for seed in seeds):
                bull = sum(1 for w in BULLISH_WORDS if w in s)
                bear = sum(1 for w in BEARISH_WORDS if w in s)
                diff = bull - bear
                topic_breakdown[topic] = _direction(diff)

        if self._trained and self._vectorizer is not None and self._classifier is not None:
            try:
                X_vec = self._vectorizer.transform([text])
                probs = self._classifier.predict_proba(X_vec)[0]
                classes = [int(c) for c in self._classifier.classes_]

                prob_map: Dict[int, float] = dict(zip(classes, probs))
                bull_p    = prob_map.get(1,  0.0)
                neutral_p = prob_map.get(0,  0.0)
                bear_p    = prob_map.get(-1, 0.0)

                pred_class = int(classes[int(np.argmax(probs))])
                direction  = LABEL_INV.get(pred_class, "NEUTRAL")
                confidence = float(max(probs))

                return DirectionalPrediction(
                    direction=direction,
                    confidence=confidence,
                    bull_prob=bull_p,
                    neutral_prob=neutral_p,
                    bear_prob=bear_p,
                    topic_breakdown=topic_breakdown,
                    input_text_preview=preview,
                )
            except Exception:
                pass

        # Rule-based fallback
        bull = sum(1 for w in BULLISH_WORDS if w in text.lower())
        bear = sum(1 for w in BEARISH_WORDS if w in text.lower())
        diff = bull - bear
        direction  = _direction(diff)
        total      = bull + bear + 1e-9
        bull_p     = bull / total
        bear_p     = bear / total
        neutral_p  = max(0.0, 1.0 - bull_p - bear_p)
        confidence = max(bull_p, bear_p, neutral_p)

        return DirectionalPrediction(
            direction=direction,
            confidence=round(confidence, 3),
            bull_prob=round(bull_p, 3),
            neutral_prob=round(neutral_p, 3),
            bear_prob=round(bear_p, 3),
            topic_breakdown=topic_breakdown,
            input_text_preview=preview,
        )

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as fh:
            pickle.dump({"vectorizer": self._vectorizer, "classifier": self._classifier}, fh)
        if self._meta:
            META_PATH.write_text(json.dumps(
                {k: v for k, v in self._meta.__dict__.items()}, indent=2
            ))

    def _load(self) -> None:
        with open(MODEL_PATH, "rb") as fh:
            bundle = pickle.load(fh)
        self._vectorizer = bundle["vectorizer"]
        self._classifier = bundle["classifier"]
        meta_data = json.loads(META_PATH.read_text())
        self._meta = ModelMeta(**meta_data)
        self._trained = True


# ── Module-level convenience ─────────────────────────────────────────────────

_SINGLETON: Optional[PHFundamentalModel] = None


def get_model(refresh: bool = False) -> PHFundamentalModel:
    """Return a trained singleton instance."""
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = PHFundamentalModel()
        _SINGLETON.train(refresh=refresh)
    return _SINGLETON


if __name__ == "__main__":
    model = PHFundamentalModel()
    meta  = model.train(refresh=True)
    print(f"Classes: {meta.classes}")
    print(f"Vocab:   {meta.vocab_size}")
    print(f"Chunks:  {meta.n_chunks}")
    print(f"CV acc:  {meta.accuracy:.2%}")

    # Quick smoke test
    samples = [
        "The Fed signalled three rate cuts are likely in 2025, boosting equities.",
        "Inflation remains sticky. Powell hawkish. Rate hikes not off the table.",
        "NFP came in line with expectations. No major market reaction expected.",
    ]
    for s in samples:
        p = model.predict(s)
        print(f"\n  Text: {s[:60]}…")
        print(f"  → {p.direction} (conf={p.confidence:.2%})")
