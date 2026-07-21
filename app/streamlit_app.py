"""
streamlit_app.py — Streamlit frontend for mental health NLP demo.

Run locally:
    streamlit run app/streamlit_app.py

Deploy to HuggingFace Spaces:
    - Upload this file as app.py in the Space root
    - Upload models/ directory
    - Set requirements.txt (see hf_requirements.txt)
"""

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import streamlit as st
from app.predict import load_models, predict_text, DISCLAIMER

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mental Health Signal Detector",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title  { font-size: 1.9rem; font-weight: 600; margin-bottom: 0.1rem; }
    .sub-title   { font-size: 0.95rem; color: #666; margin-bottom: 1.5rem; }
    .risk-box    { padding: 1rem 1.2rem; border-radius: 10px;
                   margin: 1rem 0; border-left: 5px solid; }
    .risk-high   { background: #FEF0EC; border-color: #D85A30; }
    .risk-mod    { background: #FDF6E7; border-color: #BA7517; }
    .risk-low    { background: #E8F7F2; border-color: #1D9E75; }
    .token-pill  { display: inline-block; padding: 3px 10px;
                   border-radius: 20px; font-size: 0.8rem;
                   margin: 3px 4px; font-weight: 500; }
    .token-pos   { background: #FFE8E2; color: #8B2000; }
    .token-neg   { background: #E8EEF8; color: #1A3A6B; }
    .disclaimer  { font-size: 0.78rem; color: #888; padding: 0.6rem 1rem;
                   background: #F5F5F5; border-radius: 8px;
                   border-left: 3px solid #ccc; margin-top: 1.5rem; }
    .prob-label  { font-size: 0.82rem; color: #555; margin-bottom: 2px; }
    .metric-card { background: #FAFAFA; border: 1px solid #E8E8E8;
                   border-radius: 8px; padding: 0.7rem 1rem; text-align: center; }
</style>
""", unsafe_allow_html=True)


# ── Model loading (cached) ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models...")
def get_models():
    return load_models()


# ── Helper renderers ───────────────────────────────────────────────────────────

def render_risk_box(result: dict) -> None:
    label = result['label']
    meta  = result['meta']
    css   = {'high': 'risk-high', 'moderate': 'risk-mod', 'low': 'risk-low'}[label]

    st.markdown(f"""
    <div class="risk-box {css}">
        <div style="font-size:1.3rem; font-weight:600; margin-bottom:4px;">
            {meta['emoji']} {meta['label']} &nbsp;
            <span style="font-size:0.9rem; font-weight:400; color:#555;">
              ({result['confidence']*100:.1f}% confidence)
            </span>
        </div>
        <div style="font-size:0.9rem; color:#444; margin-bottom:6px;">
            {meta['description']}
        </div>
        <div style="font-size:0.85rem; color:#333; font-weight:500;">
            💡 {meta['guidance']}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_probability_bars(probabilities: dict) -> None:
    st.markdown("**Confidence breakdown**")
    order  = ['high', 'moderate', 'low']
    colors = {'high': '#D85A30', 'moderate': '#BA7517', 'low': '#1D9E75'}
    labels = {'high': '🔴 High', 'moderate': '🟡 Moderate', 'low': '🟢 Low'}

    for cls in order:
        if cls not in probabilities:
            continue
        prob = probabilities[cls]
        st.markdown(f"<div class='prob-label'>{labels[cls]}</div>", unsafe_allow_html=True)
        st.progress(prob, text=f"{prob*100:.1f}%")


def render_token_highlights(top_tokens: list) -> None:
    if not top_tokens:
        return
    st.markdown("**Key signals detected**")
    st.caption("Tokens that most influenced this prediction:")

    pills_html = ""
    for t in top_tokens:
        css   = "token-pos" if t['direction'] == 'positive' else "token-neg"
        arrow = "▲" if t['direction'] == 'positive' else "▼"
        pills_html += f'<span class="token-pill {css}">{arrow} {t["token"]}</span>'
    st.markdown(pills_html, unsafe_allow_html=True)
    st.caption("▲ red = pushes toward predicted class  |  ▼ blue = pushes away")


# ── Main app ───────────────────────────────────────────────────────────────────

def main():
    models = get_models()

    # Header
    st.markdown('<div class="main-title">🧠 Mental Health Signal Detector</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">NLP research demo · DistilBERT + Logistic Regression · '
        'Built on self-collected survey + Kaggle data</div>',
        unsafe_allow_html=True
    )

    # Disclaimer banner
    st.warning(
        "⚠ **Research prototype only.** Not a clinical tool. "
        "If you are in crisis, please contact iCall: **9152987821** "
        "or Vandrevala Foundation: **1860-2662-345**",
        icon=None
    )

    st.divider()

    # Input area
    st.markdown("### Analyse text")
    text_input = st.text_area(
        label="Enter text to analyse",
        placeholder="Describe how your week has been, how you're feeling, or paste any free-form text...",
        height=130,
        max_chars=1000,
        label_visibility="collapsed",
    )

    col1, col2, col3 = st.columns([2, 1, 3])
    with col1:
        analyse_btn = st.button("Analyse", type="primary", use_container_width=True)
    with col2:
        clear_btn = st.button("Clear", use_container_width=True)

    if clear_btn:
        st.rerun()

    # Example texts
    with st.expander("Try an example"):
        examples = {
            "😟 High distress signal":
                "I feel completely hopeless and cannot see any reason to continue. "
                "Nothing brings me joy anymore and I feel so empty inside all the time.",
            "😊 Positive wellbeing":
                "Had a really great week. Caught up with friends, finished my project early, "
                "and feeling motivated and grateful. Sleeping well and eating right.",
            "😐 Mixed / moderate":
                "Not sure how I'm managing. Some days are okay but others feel really heavy. "
                "Struggling a bit with motivation but trying to push through.",
        }
        for label_ex, ex_text in examples.items():
            if st.button(label_ex, use_container_width=True):
                text_input = ex_text
                analyse_btn = True

    # Run inference
    if analyse_btn and text_input:
        if len(text_input.strip()) < 3:
            st.error("Please enter at least 3 characters.")
            return

        with st.spinner("Analysing..."):
            result = predict_text(text_input, models)

        if result.get('error'):
            st.error(f"Error: {result['error']}")
            return

        st.divider()
        st.markdown("### Results")

        # Risk box
        render_risk_box(result)

        # Columns: probabilities + tokens
        col_l, col_r = st.columns(2)
        with col_l:
            render_probability_bars(result['probabilities'])
        with col_r:
            render_token_highlights(result['top_tokens'])

        # Metadata
        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Inference time",  f"{result['inference_ms']} ms")
        m2.metric("Confidence",      f"{result['confidence']*100:.1f}%")
        m3.metric("Text length",     f"{len(text_input.split())} words")

        # Disclaimer
        st.markdown(
            f'<div class="disclaimer">⚠ {DISCLAIMER}</div>',
            unsafe_allow_html=True
        )

    elif analyse_btn and not text_input:
        st.warning("Please enter some text first.")

    # Sidebar: about
    with st.sidebar:
        st.markdown("### About this project")
        st.markdown("""
        **Mental Health Signal Detection**
        
        NLP pipeline built as a B.Tech final year project.
        
        **Models**
        - Logistic Regression (TF-IDF + 24 linguistic features)
        - DistilBERT fine-tuned *(coming Week 4)*
        
        **Data**
        - Kaggle: CLPsych / Depression Reddit
        - Self-collected Google Form survey (300+ responses)
        
        **Labels**
        - PHQ-2 screening instrument (0–6 scale)
        - Low (0–2) · Moderate (3–4) · High (5–6)
        
        **Explainability**
        - SHAP LinearExplainer for token importance
        - Fairness audit across demographic groups
        
        **GitHub**  
        [github.com/YOUR_USERNAME/mental-health-nlp](https://github.com)
        """)

        st.divider()
        st.markdown("### API")
        st.code("POST /predict\n{\"text\": \"your text\"}", language="json")
        st.caption("Flask REST API running alongside this demo.")

        st.divider()
        st.caption("Built with scikit-learn · HuggingFace · SHAP · Flask · Streamlit")


if __name__ == '__main__':
    main()
