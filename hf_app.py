"""
hf_app.py — HuggingFace Spaces entrypoint.

On HuggingFace Spaces, rename this file to app.py in the repo root.
HF Spaces (Streamlit SDK) looks for app.py at the root.

The models/ directory must also be at the repo root.
"""

import os
import sys

# Ensure project root is on path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# Import and run the Streamlit app
from app.streamlit_app import main

if __name__ == '__main__':
    main()
