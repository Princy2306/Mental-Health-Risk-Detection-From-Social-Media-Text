"""
api.py — Flask REST API for mental health NLP inference.

Endpoints:
    POST /predict         — classify text, return label + confidence + token highlights
    GET  /health          — liveness check
    GET  /model-info      — model metadata (classes, vocab size, version)
    POST /predict-batch   — classify multiple texts in one request (max 10)

Run locally:
    python app/api.py

Test with curl:
    curl -X POST http://localhost:5000/predict \
         -H "Content-Type: application/json" \
         -d '{"text": "I feel hopeless and cannot see any reason to continue."}'

Production deploy: Gunicorn (see Dockerfile)
    gunicorn -w 2 -b 0.0.0.0:7860 app.api:app
"""

import os
import sys
import time
import logging

# Add project root to path so `from app.predict import ...` works
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from flask import Flask, request, jsonify, abort
from app.predict import load_models, predict_text, DISCLAIMER

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S',
)
logger = logging.getLogger(__name__)

# ── App init ───────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Load models once at startup
logger.info("Loading models...")
MODELS     = load_models()
START_TIME = time.strftime('%Y-%m-%dT%H:%M:%SZ')
logger.info(f"API ready. Classes: {MODELS['le'].classes_.tolist()}")


# ── Rate limiting (simple in-memory, no Redis needed for demo) ─────────────────
REQUEST_LOG  = {}   # ip → [timestamps]
MAX_RPS      = 10   # max requests per second per IP


def _rate_check(ip: str) -> bool:
    now = time.time()
    timestamps = [t for t in REQUEST_LOG.get(ip, []) if now - t < 1.0]
    REQUEST_LOG[ip] = timestamps
    if len(timestamps) >= MAX_RPS:
        return False
    REQUEST_LOG[ip].append(now)
    return True


# ── Input validation ───────────────────────────────────────────────────────────

def _validate_text(data: dict) -> tuple[str | None, str | None]:
    """Return (text, error_message). Error is None if valid."""
    if not data or 'text' not in data:
        return None, "Missing field: 'text'"
    text = data['text']
    if not isinstance(text, str):
        return None, "'text' must be a string"
    if len(text.strip()) < 3:
        return None, "Text too short (minimum 3 characters)"
    if len(text) > 5000:
        return None, "Text too long (maximum 5000 characters)"
    return text, None


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    """Liveness check. Returns 200 if the API is running."""
    return jsonify({
        'status'    : 'ok',
        'uptime'    : START_TIME,
        'model'     : 'lr_tfidf_shap',
        'classes'   : MODELS['le'].classes_.tolist(),
        'disclaimer': DISCLAIMER,
    })


@app.route('/model-info', methods=['GET'])
def model_info():
    """Return model metadata — useful for frontend display."""
    return jsonify({
        'model_type'  : 'Logistic Regression (TF-IDF + Linguistic features)',
        'classes'     : MODELS['le'].classes_.tolist(),
        'vocab_size'  : len(MODELS['vec'].vocabulary_),
        'loaded_at'   : MODELS['loaded_at'],
        'disclaimer'  : DISCLAIMER,
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Classify a single text.

    Request body (JSON):
        {"text": "your text here"}

    Response (JSON):
        {
            "label":         "high",
            "confidence":    0.87,
            "probabilities": {"high": 0.87, "low": 0.08, "moderate": 0.05},
            "top_tokens":    [{"token": "hopeless", "shap": 0.23, "direction": "positive"}, ...],
            "inference_ms":  1.2,
            "meta": {
                "label":       "High Risk",
                "emoji":       "🔴",
                "color":       "#D85A30",
                "description": "...",
                "guidance":    "..."
            },
            "disclaimer": "..."
        }
    """
    # Rate limiting
    ip = request.remote_addr or '127.0.0.1'
    if not _rate_check(ip):
        return jsonify({'error': 'Rate limit exceeded. Max 10 requests/second.'}), 429

    # Parse and validate
    data         = request.get_json(silent=True) or {}
    text, err    = _validate_text(data)
    if err:
        return jsonify({'error': err}), 400

    # Inference
    result = predict_text(text, MODELS)
    if result.get('error'):
        return jsonify({'error': result['error']}), 400

    logger.info(f"[/predict] label={result['label']} conf={result['confidence']:.3f} "
                f"ms={result['inference_ms']} ip={ip}")

    # Remove internal fields before returning
    result.pop('error', None)
    return jsonify(result)


@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    """
    Classify up to 10 texts in one request.

    Request body:
        {"texts": ["text1", "text2", ...]}

    Response:
        {"results": [<predict response>, ...]}
    """
    ip = request.remote_addr or '127.0.0.1'
    if not _rate_check(ip):
        return jsonify({'error': 'Rate limit exceeded.'}), 429

    data = request.get_json(silent=True) or {}
    if 'texts' not in data or not isinstance(data['texts'], list):
        return jsonify({'error': "Missing field: 'texts' (must be a list)"}), 400
    if len(data['texts']) > 10:
        return jsonify({'error': 'Maximum 10 texts per batch request.'}), 400

    results = []
    for text in data['texts']:
        text_str, err = _validate_text({'text': text})
        if err:
            results.append({'error': err, 'label': None})
        else:
            r = predict_text(text_str, MODELS)
            r.pop('error', None)
            results.append(r)

    return jsonify({'results': results, 'count': len(results)})


# ── Error handlers ─────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found',
                    'available': ['/predict', '/predict-batch', '/health', '/model-info']}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({'error': f'Method not allowed on this endpoint'}), 405


@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal error: {e}")
    return jsonify({'error': 'Internal server error. Please try again.'}), 500


# ── Dev server ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    logger.info(f"Starting Flask dev server on port {port}")
    logger.info(f"Test: curl -X POST http://localhost:{port}/predict "
                f"-H 'Content-Type: application/json' "
                f"-d '{{\"text\": \"I feel hopeless\"}}'")
    app.run(host='0.0.0.0', port=port, debug=False)
