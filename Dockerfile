# Dockerfile — Mental Health NLP Demo
# Builds a single image running the Streamlit frontend.
# Used by HuggingFace Spaces (Dockerfile SDK).
#
# Build locally:
#   docker build -t mh-nlp-demo .
#   docker run -p 7860:7860 mh-nlp-demo
#
# Access at: http://localhost:7860

FROM python:3.10-slim

# ── System deps ────────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────────
WORKDIR /app

# ── Python deps ────────────────────────────────────────────────────────────────
# Copy requirements first (layer caching — only reinstalls if reqs change)
COPY hf_requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ── App code ───────────────────────────────────────────────────────────────────
COPY . .

# ── HuggingFace Spaces: must run on port 7860 ─────────────────────────────────
EXPOSE 7860

# Create a non-root user (security best practice)
RUN useradd -m -u 1000 appuser
RUN chown -R appuser:appuser /app
USER appuser

# ── Entrypoint ─────────────────────────────────────────────────────────────────
# HF Spaces looks for app.py by convention — we symlink from app/streamlit_app.py
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
