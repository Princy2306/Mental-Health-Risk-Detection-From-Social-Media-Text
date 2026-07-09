"""
features.py — Linguistic feature extraction for mental health NLP project.

Used in:
  - notebooks/02_features.ipynb  (exploration)
  - train.py                     (modelling pipeline)
  - app/predict.py               (inference)

All functions are pure and stateless — safe to import anywhere.
"""

import re
import numpy as np
import pandas as pd
from typing import Dict

# ── Regex patterns (compiled once at module load) ──────────────────────────────

NEGATION = re.compile(
    r'\b(not|never|no|neither|nor|cannot|can\'t|won\'t|don\'t|doesn\'t|'
    r'didn\'t|isn\'t|wasn\'t|aren\'t|weren\'t|nothing|nobody|nowhere|'
    r'hardly|barely|scarcely)\b',
    re.IGNORECASE
)

FP_SINGULAR = re.compile(r'\b(i|me|my|myself|mine)\b', re.IGNORECASE)
FP_PLURAL   = re.compile(r'\b(we|us|our|ourselves|ours)\b', re.IGNORECASE)
QUESTION    = re.compile(r'\?')
EXCLAIM     = re.compile(r'!')
ELLIPSIS    = re.compile(r'\.\.\.')
PUNCT       = re.compile(r'[^\w\s]')
SENTENCE_SEP = re.compile(r'[.!?]+')

# Emotionally charged word lists (hand-curated, research-backed)
POSITIVE_AFFECT = re.compile(
    r'\b(happy|joy|great|wonderful|love|excited|grateful|hopeful|'
    r'cheerful|delighted|content|optimistic|energetic|motivated|proud|'
    r'peaceful|calm|relaxed|glad|pleased|thrilled|fantastic|amazing)\b',
    re.IGNORECASE
)

NEGATIVE_AFFECT = re.compile(
    r'\b(sad|hopeless|worthless|empty|alone|lonely|anxious|depressed|'
    r'miserable|terrible|awful|horrible|dreadful|desperate|helpless|'
    r'exhausted|numb|broken|lost|afraid|scared|worried|overwhelmed|'
    r'burden|hate|hate|fail|failure|useless|pointless|meaningless)\b',
    re.IGNORECASE
)

FUTURE_TENSE = re.compile(
    r'\b(will|shall|going to|gonna|would|might|could|may|plan to|want to|'
    r'hope to|expect to|intend to)\b',
    re.IGNORECASE
)

PAST_TENSE_MARKER = re.compile(r'\b\w+(ed|was|were|had|did)\b', re.IGNORECASE)


# ── Helper ─────────────────────────────────────────────────────────────────────

def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


# ── Core feature extractor ─────────────────────────────────────────────────────

def extract_linguistic_features(text: str) -> Dict[str, float]:
    """
    Extract 24 hand-crafted linguistic features from a single text string.

    Returns a flat dict of float values — compatible with pd.DataFrame.

    Features cover:
        Surface stats     : length, word count, sentence count, avg word/sent length
        Negation          : count + rate
        Pronouns          : 1st-person singular/plural counts + rates
        Affect words      : positive vs negative lexicon matches
        Temporal focus    : future vs past tense markers
        Richness          : type-token ratio (vocabulary diversity)
        Punctuation       : question marks, exclamations, ellipsis, density
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {k: 0.0 for k in _FEATURE_KEYS}

    text = text.strip()
    words = text.split()
    word_count = len(words)
    char_len = len(text)

    sentences = [s.strip() for s in SENTENCE_SEP.split(text) if len(s.strip()) > 2]
    sentence_count = max(len(sentences), 1)
    unique_words = set(w.lower() for w in words)

    negation_count    = len(NEGATION.findall(text))
    fp_singular_count = len(FP_SINGULAR.findall(text))
    fp_plural_count   = len(FP_PLURAL.findall(text))
    positive_count    = len(POSITIVE_AFFECT.findall(text))
    negative_count    = len(NEGATIVE_AFFECT.findall(text))
    future_count      = len(FUTURE_TENSE.findall(text))
    past_count        = len(PAST_TENSE_MARKER.findall(text))
    punct_count       = len(PUNCT.findall(text))

    return {
        # Surface
        'char_len'            : float(char_len),
        'word_count'          : float(word_count),
        'avg_word_len'        : _safe_div(sum(len(w) for w in words), word_count),
        'sentence_count'      : float(sentence_count),
        'avg_sent_len'        : _safe_div(word_count, sentence_count),

        # Negation
        'negation_count'      : float(negation_count),
        'negation_rate'       : _safe_div(negation_count, word_count),

        # Pronouns
        'fp_singular_count'   : float(fp_singular_count),
        'fp_singular_rate'    : _safe_div(fp_singular_count, word_count),
        'fp_plural_count'     : float(fp_plural_count),
        'fp_plural_rate'      : _safe_div(fp_plural_count, word_count),

        # Affect
        'positive_affect'     : float(positive_count),
        'negative_affect'     : float(negative_count),
        'affect_balance'      : float(positive_count - negative_count),
        'affect_ratio'        : _safe_div(positive_count, max(negative_count, 1)),

        # Temporal
        'future_tense_count'  : float(future_count),
        'past_tense_count'    : float(past_count),
        'temporal_balance'    : float(future_count - past_count),

        # Richness
        'type_token_ratio'    : _safe_div(len(unique_words), word_count),

        # Punctuation
        'question_count'      : float(len(QUESTION.findall(text))),
        'exclamation_count'   : float(len(EXCLAIM.findall(text))),
        'ellipsis_count'      : float(len(ELLIPSIS.findall(text))),
        'punctuation_count'   : float(punct_count),
        'punctuation_density' : _safe_div(punct_count, char_len),
    }


# Derive feature key list from a dummy call (used for zero-fill on empty text)
_FEATURE_KEYS = list(extract_linguistic_features("dummy text for key extraction").keys())


def extract_features_batch(texts: pd.Series) -> pd.DataFrame:
    """
    Apply extract_linguistic_features to a full pandas Series.
    Returns a DataFrame with one row per text.

    Usage:
        feat_df = extract_features_batch(df['text'])
    """
    return pd.DataFrame(texts.apply(extract_linguistic_features).tolist())


# ── Text cleaning ──────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Minimal text cleaning — preserves meaning-bearing punctuation.
    Does NOT lowercase (BERT tokenizer handles case internally).
    """
    if not isinstance(text, str):
        return ""
    text = text.strip()
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '[URL]', text)
    # Remove usernames (Reddit-style u/username)
    text = re.sub(r'u/\w+', '[USER]', text)
    # Remove subreddit tags
    text = re.sub(r'r/\w+', '[SUB]', text)
    return text


def clean_texts_batch(texts: pd.Series) -> pd.Series:
    """Apply clean_text to a full Series."""
    return texts.apply(clean_text)


# ── PSS reverse scoring (for survey data) ─────────────────────────────────────

def compute_pss_score(control: float, difficulties: float, confidence: float) -> float:
    """
    Compute PSS-3 composite score.
    'confidence' is reverse-scored (1→5, 2→4, 3→3, 4→2, 5→1).
    All inputs should be on 1–5 scale.
    """
    confidence_reversed = 6.0 - confidence
    return control + difficulties + confidence_reversed


def phq2_to_label(score: int) -> str:
    """Map PHQ-2 sum score to 3-class label."""
    if score <= 2:
        return 'low'
    elif score <= 4:
        return 'moderate'
    else:
        return 'high'


if __name__ == '__main__':
    # Quick smoke test
    sample = "I feel so hopeless and cannot see any point. Nothing ever gets better. Why bother?"
    feats = extract_linguistic_features(sample)
    print("Feature extraction smoke test:")
    for k, v in feats.items():
        print(f"  {k:<25} {v:.4f}")
    print(f"\nTotal features: {len(feats)}")
