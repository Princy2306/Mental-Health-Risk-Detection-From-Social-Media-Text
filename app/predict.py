"""
predict.py — Core inference module for mental health NLP project.

Loaded once at app startup. Used by both Flask API and Streamlit frontend.

Usage:
    from predict import load_models, predict_text, get_risk_guidance

    models = load_models()
    result = predict_text("I feel hopeless and empty", models)
    print(result)
"""

import os
import time
import warnings
import numpy as np
import joblib

warnings.filterwarnings('ignore')

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
MODEL_DIR = os.path.join(ROOT, 'models')

# Risk level metadata — shown in UI and API response
RISK_META = {
    'low': {
        'label'      : 'Low Risk',
        'emoji'      : '🟢',
        'color'      : '#1D9E75',
        'description': 'The text suggests generally positive or neutral wellbeing signals.',
        'guidance'   : 'Keep up healthy habits. Reach out to friends and stay connected.',
    },
    'moderate': {
        'label'      : 'Moderate Risk',
        'emoji'      : '🟡',
        'color'      : '#BA7517',
        'description': 'The text contains some signals of stress or emotional difficulty.',
        'guidance'   : 'Consider talking to a friend, counsellor, or trusted person. '
                       'Small steps like sleep, movement, and social contact help.',
    },
    'high': {
        'label'      : 'High Risk',
        'emoji'      : '🔴',
        'color'      : '#D85A30',
        'description': 'The text contains signals associated with significant distress.',
        'guidance'   : 'Please reach out for support. '
                       'iCall (India): 9152987821 | Vandrevala Foundation: 1860-2662-345 | '
                       'Speak to your college counsellor.',
    },
}

DISCLAIMER = (
    "⚠ This tool is a research prototype, not a clinical instrument. "
    "It must not be used for diagnosis or treatment decisions. "
    "If you or someone you know is in crisis, please contact a mental health professional."
)


def load_models(model_dir: str = MODEL_DIR) -> dict:
    """
    Load all model artefacts from disk.
    Call once at app startup — not on every request.

    Returns dict with keys: le, vec, lr, shap_explainer (optional)
    """
    le  = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
    vec = joblib.load(os.path.join(model_dir, 'tfidf_vectorizer.pkl'))
    lr  = joblib.load(os.path.join(model_dir, 'lr_tfidf_shap.pkl'))

    # Load SHAP explainer if available (built in Week 4)
    shap_explainer = None
    shap_path = os.path.join(model_dir, 'shap_explainer.pkl')
    if os.path.exists(shap_path):
        try:
            shap_explainer = joblib.load(shap_path)
        except Exception:
            pass  # SHAP optional — app still works without it

    print(f"[predict] Models loaded. Classes: {le.classes_.tolist()}")
    return {
        'le'            : le,
        'vec'           : vec,
        'lr'            : lr,
        'shap_explainer': shap_explainer,
        'loaded_at'     : time.strftime('%Y-%m-%dT%H:%M:%SZ'),
    }


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def predict_text(text: str, models: dict) -> dict:
    """
    Run inference on a single text string.

    Parameters
    ----------
    text   : raw input string (any length — truncated internally)
    models : output of load_models()

    Returns
    -------
    dict with keys:
        label         — predicted class string ('low'/'moderate'/'high')
        confidence    — float 0–1, probability of predicted class
        probabilities — dict {class: probability}
        top_tokens    — list of dicts [{token, shap, direction}] if SHAP available
        inference_ms  — float, wall-clock inference time
        meta          — risk metadata dict (label, emoji, color, description, guidance)
        disclaimer    — str
        error         — str or None
    """
    le  = models['le']
    vec = models['vec']
    lr  = models['lr']

    # Input validation
    if not isinstance(text, str) or len(text.strip()) < 3:
        return {'error': 'Text must be at least 3 characters.', 'label': None}

    text = text.strip()[:1000]  # hard cap — DistilBERT limit; also prevents abuse

    t0 = time.perf_counter()

    x       = vec.transform([text])
    pred_idx = int(lr.predict(x)[0])

    # Probabilities: use predict_proba if available, else softmax of decision_function
    try:
        proba = lr.predict_proba(x)[0]
    except AttributeError:
        proba = _softmax(lr.decision_function(x)[0])

    label      = str(le.classes_[pred_idx])
    confidence = float(proba[pred_idx])
    all_proba  = {
        str(le.classes_[i]): round(float(p), 4)
        for i, p in enumerate(proba)
    }

    # SHAP token highlights (optional)
    top_tokens = []
    if models.get('shap_explainer') is not None:
        try:
            from shap_utils import get_token_highlights
            top_tokens = get_token_highlights(
                text, models['shap_explainer'], vec, top_n=5
            )
        except Exception:
            pass

    # Fallback token highlights via model coefficients (always available)
    if not top_tokens:
        top_tokens = _coef_token_highlights(text, x, lr, vec, le, pred_idx, n=5)

    inference_ms = round((time.perf_counter() - t0) * 1000, 2)

    return {
        'label'        : label,
        'confidence'   : round(confidence, 4),
        'probabilities': all_proba,
        'top_tokens'   : top_tokens,
        'inference_ms' : inference_ms,
        'meta'         : RISK_META[label],
        'disclaimer'   : DISCLAIMER,
        'error'        : None,
    }


def _coef_token_highlights(text, x_sparse, lr, vec, le, pred_idx, n=5):
    """
    Coefficient-based token highlights — fast fallback when SHAP is unavailable.
    Returns the n tokens present in the text with the largest coefficient magnitude
    for the predicted class.
    """
    try:
        coef      = lr.coef_[pred_idx]             # (n_features,)
        feat_names = vec.get_feature_names_out()
        nonzero   = x_sparse.nonzero()[1]           # indices of tokens in this text
        if len(nonzero) == 0:
            return []

        token_coef = [(feat_names[i], float(coef[i])) for i in nonzero]
        token_coef.sort(key=lambda t: abs(t[1]), reverse=True)

        return [
            {
                'token'    : tok,
                'shap'     : round(val, 5),
                'direction': 'positive' if val > 0 else 'negative',
            }
            for tok, val in token_coef[:n]
        ]
    except Exception:
        return []


def get_risk_guidance(label: str) -> dict:
    """Return UI-facing risk metadata for a given class label."""
    return RISK_META.get(label, RISK_META['moderate'])


if __name__ == '__main__':
    print("Loading models...")
    models = load_models()

    samples = [
        "I feel completely hopeless and cannot see any reason to continue with anything.",
        "Had a really productive day, feeling energetic and motivated about my goals!",
        "Struggling a bit this week, not sure how to handle everything piling up.",
    ]

    print("\nInference test:")
    for text in samples:
        r = predict_text(text, models)
        print(f"\n  Text     : {text[:60]}...")
        print(f"  Label    : {r['meta']['emoji']} {r['label']} ({r['confidence']:.3f})")
        print(f"  Tokens   : {[(t['token'], t['direction']) for t in r['top_tokens'][:3]]}")
        print(f"  Latency  : {r['inference_ms']} ms")
