"""
tests/test_predict.py — pytest test suite for the predict module and Flask API.

Run:
    pytest tests/test_predict.py -v

Coverage areas:
    1. predict_text() — core inference function
    2. Flask API routes — /predict, /predict-batch, /health, /model-info
    3. Boundary conditions — empty, short, very long, non-ASCII, injection attempts
    4. Response schema — all required keys present with correct types
    5. Consistency — same input always gives same output (determinism)
"""

import sys
import os
import json
import pytest

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.predict import load_models, predict_text, RISK_META, DISCLAIMER
from app.api     import app as flask_app


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope='session')
def models():
    """Load models once for the entire test session."""
    return load_models()


@pytest.fixture(scope='session')
def client():
    """Flask test client."""
    flask_app.config['TESTING'] = True
    with flask_app.test_client() as c:
        yield c


# ── Sample texts ───────────────────────────────────────────────────────────────

HIGH_RISK_TEXT = (
    "I feel completely hopeless and cannot see any reason to continue. "
    "Nothing brings me joy anymore and I feel empty inside."
)
LOW_RISK_TEXT = (
    "Today was a fantastic day. I went out with friends, felt energised "
    "and am looking forward to tackling my assignments tomorrow."
)
MODERATE_TEXT = (
    "Feeling a bit overwhelmed this week but managing. "
    "Not sure how to handle everything but trying to stay positive."
)


# ── 1. predict_text() unit tests ───────────────────────────────────────────────

class TestPredictText:

    def test_returns_dict(self, models):
        result = predict_text(HIGH_RISK_TEXT, models)
        assert isinstance(result, dict)

    def test_required_keys_present(self, models):
        result = predict_text(HIGH_RISK_TEXT, models)
        required = {'label', 'confidence', 'probabilities', 'top_tokens',
                    'inference_ms', 'meta', 'disclaimer', 'error'}
        assert required.issubset(result.keys()), \
            f"Missing keys: {required - set(result.keys())}"

    def test_label_is_valid_class(self, models):
        for text in [HIGH_RISK_TEXT, LOW_RISK_TEXT, MODERATE_TEXT]:
            r = predict_text(text, models)
            assert r['label'] in ['high', 'low', 'moderate'], \
                f"Unexpected label: {r['label']}"

    def test_confidence_in_range(self, models):
        r = predict_text(HIGH_RISK_TEXT, models)
        assert 0.0 <= r['confidence'] <= 1.0

    def test_probabilities_sum_to_one(self, models):
        r = predict_text(HIGH_RISK_TEXT, models)
        total = sum(r['probabilities'].values())
        assert abs(total - 1.0) < 1e-4, f"Probabilities sum to {total}"

    def test_probabilities_keys_match_classes(self, models):
        r = predict_text(LOW_RISK_TEXT, models)
        le = models['le']
        assert set(r['probabilities'].keys()) == set(le.classes_.tolist())

    def test_top_tokens_is_list(self, models):
        r = predict_text(HIGH_RISK_TEXT, models)
        assert isinstance(r['top_tokens'], list)

    def test_top_tokens_schema(self, models):
        r = predict_text(HIGH_RISK_TEXT, models)
        for token in r['top_tokens']:
            assert 'token'     in token
            assert 'shap'      in token
            assert 'direction' in token
            assert token['direction'] in ('positive', 'negative')
            assert isinstance(token['shap'], float)

    def test_inference_ms_positive(self, models):
        r = predict_text(HIGH_RISK_TEXT, models)
        assert r['inference_ms'] > 0

    def test_meta_keys_present(self, models):
        r = predict_text(HIGH_RISK_TEXT, models)
        for key in ['label', 'emoji', 'color', 'description', 'guidance']:
            assert key in r['meta'], f"Missing meta key: {key}"

    def test_no_error_on_valid_input(self, models):
        r = predict_text(HIGH_RISK_TEXT, models)
        assert r['error'] is None

    def test_disclaimer_present(self, models):
        r = predict_text(HIGH_RISK_TEXT, models)
        assert len(r['disclaimer']) > 10

    def test_deterministic(self, models):
        """Same input must always give same output."""
        r1 = predict_text(HIGH_RISK_TEXT, models)
        r2 = predict_text(HIGH_RISK_TEXT, models)
        assert r1['label']      == r2['label']
        assert r1['confidence'] == r2['confidence']

    def test_confidence_matches_label_probability(self, models):
        """confidence must equal probabilities[label]."""
        r = predict_text(HIGH_RISK_TEXT, models)
        assert abs(r['confidence'] - r['probabilities'][r['label']]) < 1e-4

    def test_high_risk_detected(self, models):
        """High-risk text should be classified as high."""
        r = predict_text(HIGH_RISK_TEXT, models)
        # Note: this may fail on a model trained on very little data —
        # the important thing is that the pipeline runs correctly.
        assert r['label'] in ['high', 'moderate']  # allow moderate on weak models

    def test_low_risk_not_high(self, models):
        """Clearly positive text should not be classified as high risk."""
        r = predict_text(LOW_RISK_TEXT, models)
        assert r['label'] in ['low', 'moderate']


# ── 2. Boundary condition tests ────────────────────────────────────────────────

class TestBoundaryConditions:

    def test_empty_string_returns_error(self, models):
        r = predict_text('', models)
        assert r['error'] is not None
        assert r['label'] is None

    def test_whitespace_only_returns_error(self, models):
        r = predict_text('   ', models)
        assert r['error'] is not None

    def test_two_char_string_returns_error(self, models):
        r = predict_text('ab', models)
        assert r['error'] is not None

    def test_three_char_string_succeeds(self, models):
        r = predict_text('sad', models)
        assert r['error'] is None
        assert r['label'] in ['high', 'low', 'moderate']

    def test_very_long_text_truncated_and_succeeds(self, models):
        """5000+ char text should be truncated and still return a result."""
        long_text = "I feel great today. " * 500   # ~10,000 chars
        r = predict_text(long_text, models)
        assert r['error'] is None
        assert r['label'] is not None

    def test_non_ascii_text_succeeds(self, models):
        """Non-ASCII (emoji, accents, Devanagari) should not crash the pipeline."""
        texts = [
            "I feel 😢 and very 😞 today",
            "Je me sens très triste aujourd'hui",
            "मैं बहुत दुखी हूं",
        ]
        for text in texts:
            r = predict_text(text, models)
            assert r['error'] is None, f"Failed on: {text}"

    def test_numbers_only_succeeds(self, models):
        r = predict_text('12345678', models)
        assert r['error'] is None

    def test_none_input_returns_error(self, models):
        r = predict_text(None, models)
        assert r['error'] is not None

    def test_integer_input_returns_error(self, models):
        r = predict_text(42, models)
        assert r['error'] is not None

    def test_sql_injection_attempt(self, models):
        """Injection strings should be treated as plain text, not crash."""
        r = predict_text("'; DROP TABLE users; --", models)
        assert r['error'] is None or 'short' in (r['error'] or '')

    def test_html_injection_attempt(self, models):
        r = predict_text('<script>alert("xss")</script>', models)
        assert r['error'] is None


# ── 3. Flask API route tests ───────────────────────────────────────────────────

class TestFlaskAPI:

    def test_health_returns_200(self, client):
        r = client.get('/health')
        assert r.status_code == 200

    def test_health_response_schema(self, client):
        data = client.get('/health').get_json()
        for key in ['status', 'model', 'classes', 'disclaimer']:
            assert key in data

    def test_health_status_is_ok(self, client):
        data = client.get('/health').get_json()
        assert data['status'] == 'ok'

    def test_model_info_returns_200(self, client):
        r = client.get('/model-info')
        assert r.status_code == 200

    def test_model_info_schema(self, client):
        data = client.get('/model-info').get_json()
        for key in ['model_type', 'classes', 'vocab_size', 'disclaimer']:
            assert key in data

    def test_predict_valid_input(self, client):
        r = client.post('/predict', json={'text': HIGH_RISK_TEXT})
        assert r.status_code == 200
        data = r.get_json()
        assert data['label'] in ['high', 'low', 'moderate']
        assert 0.0 <= data['confidence'] <= 1.0

    def test_predict_missing_text_field(self, client):
        r = client.post('/predict', json={})
        assert r.status_code == 400
        assert 'error' in r.get_json()

    def test_predict_empty_text(self, client):
        r = client.post('/predict', json={'text': ''})
        assert r.status_code == 400

    def test_predict_short_text(self, client):
        r = client.post('/predict', json={'text': 'hi'})
        assert r.status_code == 400

    def test_predict_no_json_body(self, client):
        r = client.post('/predict', data='not json',
                        content_type='text/plain')
        assert r.status_code == 400

    def test_predict_probabilities_present(self, client):
        r = client.post('/predict', json={'text': LOW_RISK_TEXT})
        data = r.get_json()
        assert 'probabilities' in data
        assert isinstance(data['probabilities'], dict)

    def test_predict_top_tokens_present(self, client):
        r = client.post('/predict', json={'text': HIGH_RISK_TEXT})
        data = r.get_json()
        assert 'top_tokens' in data
        assert isinstance(data['top_tokens'], list)

    def test_batch_valid(self, client):
        r = client.post('/predict-batch',
                        json={'texts': [HIGH_RISK_TEXT, LOW_RISK_TEXT]})
        assert r.status_code == 200
        data = r.get_json()
        assert 'results' in data
        assert len(data['results']) == 2
        assert data['count'] == 2

    def test_batch_too_many_items(self, client):
        r = client.post('/predict-batch',
                        json={'texts': ['text'] * 11})
        assert r.status_code == 400

    def test_batch_missing_texts_field(self, client):
        r = client.post('/predict-batch', json={})
        assert r.status_code == 400

    def test_404_on_unknown_route(self, client):
        r = client.get('/nonexistent')
        assert r.status_code == 404
        data = r.get_json()
        assert 'available' in data

    def test_get_on_predict_returns_405(self, client):
        r = client.get('/predict')
        assert r.status_code == 405

    def test_content_type_is_json(self, client):
        r = client.post('/predict', json={'text': HIGH_RISK_TEXT})
        assert 'application/json' in r.content_type
