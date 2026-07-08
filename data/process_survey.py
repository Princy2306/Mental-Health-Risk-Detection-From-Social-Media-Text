"""
Week 1/2 — Google Form survey response processor.

Usage:
    1. Export your Google Form responses as CSV (Responses tab → Download CSV).
    2. Save as: data/raw/survey_responses.csv
    3. Run: python data/process_survey.py

This script:
    - Drops all PII columns
    - Renames columns to short snake_case keys
    - Computes PHQ-2 score and 3-class label
    - Validates response quality (min text length)
    - Saves anonymised output to data/processed/survey_clean.csv
"""

import os
import re
import pandas as pd

RAW_PATH = "data/raw/survey_responses.csv"
OUT_PATH = "data/processed/survey_clean.csv"

# ── Column name mapping ────────────────────────────────────────────────────────
# Left  = exact column header from your Google Form export
# Right = short internal name used throughout the project
# UPDATE LEFT SIDE to match your actual form column headers.
COLUMN_MAP = {
    "Timestamp": "timestamp",
    "Age range": "age",
    "Gender": "gender",
    "Year of study": "year",
    "Current academic semester phase": "semester_phase",

    # Open text — NLP inputs
    "Describe how your week has been going, in 3–5 sentences. Write freely.": "text_week",
    "What's been on your mind the most this week?": "text_mind",
    "How would you describe your social life right now?": "text_social",

    # PHQ-2 items (0=Not at all, 1=Several days, 2=More than half, 3=Nearly every day)
    "Little interest or pleasure in doing things": "phq_interest",
    "Feeling down, depressed, or hopeless": "phq_depressed",

    # Lifestyle features
    "Average sleep last week": "sleep",
    "How many times did you leave your room/home socially this week?": "social_outings",
    "Daily screen time estimate": "screen_time",
    "Cups of coffee/tea per day": "caffeine",
    "Have you exercised at least once this week?": "exercise",
    "Meals per day on average this week": "meals",

    # PSS-3 (1=Never, 5=Very Often)
    "I have felt unable to control important things in my life": "pss_control",
    "I have felt difficulties were piling up so high I could not overcome them": "pss_difficulties",
    "I have felt confident about my ability to handle personal problems": "pss_confidence",

    # Optional closing text
    "Is there anything else you'd like to share about how you're feeling? (completely optional)": "text_extra",
}

# PHQ-2 ordinal encoding — update if your form uses different labels
PHQ_MAP = {
    "Not at all": 0,
    "Several days": 1,
    "More than half the days": 2,
    "Nearly every day": 3,
}

# Lifestyle ordinal encodings
SLEEP_MAP = {"< 5 hrs": 1, "5–6": 2, "6–7": 3, "7–8": 4, "8+": 5}
SOCIAL_MAP = {"0": 0, "1–2": 1, "3–5": 3, "5+": 5}
SCREEN_MAP = {"< 2 hrs": 1, "2–4": 2, "4–6": 3, "6+": 4}
CAFFEINE_MAP = {"0": 0, "1–2": 1, "3–4": 3, "5+": 5}
MEALS_MAP = {"1": 1, "2": 2, "3": 3, "irregular": 2}

# PSS confidence is REVERSE SCORED (high confidence = low stress)
PSS_REVERSE = {"1": 5, "2": 4, "3": 3, "4": 2, "5": 1}


def phq2_to_label(score: int) -> str:
    if score <= 2:
        return "low"
    elif score <= 4:
        return "moderate"
    else:
        return "high"


def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def main():
    if not os.path.exists(RAW_PATH):
        print(f"[MISSING] {RAW_PATH}")
        print("Export your Google Form responses as CSV and save there.")
        return

    df = pd.read_csv(RAW_PATH)
    print(f"Raw responses loaded: {df.shape}")
    print(f"Columns found: {list(df.columns)}\n")

    # ── Rename columns ─────────────────────────────────────────────────────────
    present_cols = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    missing_cols = [k for k in COLUMN_MAP if k not in df.columns]

    if missing_cols:
        print("[WARNING] These expected columns were NOT found — update COLUMN_MAP:")
        for c in missing_cols:
            print(f"  '{c}'")

    df = df.rename(columns=present_cols)

    # ── Drop PII ───────────────────────────────────────────────────────────────
    pii_cols = ["Email Address", "email", "Name", "name", "Student ID", "Phone"]
    df = df.drop(columns=[c for c in pii_cols if c in df.columns], errors="ignore")
    df = df.drop(columns=["timestamp"], errors="ignore")

    # ── Encode PHQ-2 ──────────────────────────────────────────────────────────
    for col in ["phq_interest", "phq_depressed"]:
        if col in df.columns:
            df[col] = df[col].map(PHQ_MAP)

    if "phq_interest" in df.columns and "phq_depressed" in df.columns:
        df["phq2_score"] = df["phq_interest"].fillna(0) + df["phq_depressed"].fillna(0)
        df["label"] = df["phq2_score"].apply(phq2_to_label)
        print(f"PHQ-2 label distribution:\n{df['label'].value_counts()}\n")

    # ── Encode lifestyle features ──────────────────────────────────────────────
    encode_pairs = [
        ("sleep", SLEEP_MAP),
        ("social_outings", SOCIAL_MAP),
        ("screen_time", SCREEN_MAP),
        ("caffeine", CAFFEINE_MAP),
        ("meals", MEALS_MAP),
    ]
    for col, mapping in encode_pairs:
        if col in df.columns:
            df[col + "_enc"] = df[col].map(mapping)

    if "exercise" in df.columns:
        df["exercise_enc"] = df["exercise"].map({"Yes": 1, "No": 0})

    # ── Encode PSS-3 ──────────────────────────────────────────────────────────
    for col in ["pss_control", "pss_difficulties"]:
        if col in df.columns:
            df[col + "_enc"] = df[col].astype(str).map(
                {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5}
            )
    if "pss_confidence" in df.columns:
        df["pss_confidence_enc"] = df["pss_confidence"].astype(str).map(PSS_REVERSE)

    if all(c in df.columns for c in ["pss_control_enc", "pss_difficulties_enc", "pss_confidence_enc"]):
        df["pss_score"] = (
            df["pss_control_enc"].fillna(3)
            + df["pss_difficulties_enc"].fillna(3)
            + df["pss_confidence_enc"].fillna(3)
        )

    # ── Clean text columns ────────────────────────────────────────────────────
    for col in ["text_week", "text_mind", "text_social", "text_extra"]:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)

    # ── Combine text fields into one NLP input ────────────────────────────────
    text_cols = [c for c in ["text_week", "text_mind", "text_social"] if c in df.columns]
    df["text_combined"] = df[text_cols].apply(
        lambda row: " ".join([v for v in row if isinstance(v, str) and len(v) > 5]),
        axis=1
    )

    # ── Quality filter ────────────────────────────────────────────────────────
    before = len(df)
    df = df[df["text_combined"].str.len() >= 30]
    print(f"Dropped {before - len(df)} rows with insufficient text (< 30 chars)")

    # ── Drop rows missing label ───────────────────────────────────────────────
    if "label" in df.columns:
        df = df.dropna(subset=["label"])

    df = df.reset_index(drop=True)

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"\n[SAVED] {OUT_PATH}")
    print(f"Final shape: {df.shape}")
    print(f"\nColumns in output:\n{list(df.columns)}")
    if "label" in df.columns:
        print(f"\nLabel counts:\n{df['label'].value_counts()}")


if __name__ == "__main__":
    main()
