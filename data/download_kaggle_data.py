"""
Week 1 — Kaggle dataset download + validation script.

Datasets to download manually from Kaggle:
  1. Depression Reddit dataset:
     https://www.kaggle.com/datasets/infamouscoder/depression-reddit-cleaned
     → Save as: data/raw/depression_reddit.csv

  2. Mental health corpus (CLPsych-style):
     https://www.kaggle.com/datasets/reihanenamdari/mental-health-corpus
     → Save as: data/raw/mental_health_corpus.csv

Run after placing CSVs in data/raw/:
    python data/download_kaggle_data.py
"""

import os
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


# ── Dataset specs ──────────────────────────────────────────────────────────────
DATASETS = {
    "depression_reddit": {
        "filename": "depression_reddit.csv",
        "text_col": "clean_text",
        "label_col": "is_depression",
        "label_map": {0: "low", 1: "high"},
        "expected_rows": 7000,
    },
    "mental_health_corpus": {
    "filename": "mental_health_corpus.csv",
    "text_col": "text",
    "label_col": "label",
    "label_map": {
        0: "low",
        1: "high",
    },
    "expected_rows": 1000,
},
}


def load_and_validate(name: str, spec: dict) -> pd.DataFrame | None:
    path = os.path.join(RAW_DIR, spec["filename"])

    if not os.path.exists(path):
        print(f"[MISSING] {spec['filename']} — download from Kaggle and place in data/raw/")
        return None

    df = pd.read_csv(path)
    print(f"\n{'='*55}")
    print(f"Dataset : {name}")
    print(f"Shape   : {df.shape}")
    print(f"Columns : {list(df.columns)}")
    print(f"Nulls   :\n{df.isnull().sum()}")

    # Check expected columns exist
    for col in [spec["text_col"], spec["label_col"]]:
        if col not in df.columns:
            print(f"[ERROR] Expected column '{col}' not found. Check column names above.")
            return None

    print(f"\nLabel distribution:\n{df[spec['label_col']].value_counts()}")
    print(f"\nSample text:\n{df[spec['text_col']].iloc[0][:200]}")

    # Row count sanity check
    if len(df) < spec["expected_rows"] * 0.5:
        print(f"[WARNING] Only {len(df)} rows — expected ~{spec['expected_rows']}. Verify download.")

    return df


def standardise(df: pd.DataFrame, spec: dict, source_name: str) -> pd.DataFrame:
    """Rename columns to unified schema: text, label, source."""
    out = pd.DataFrame()
    out["text"] = df[spec["text_col"]].astype(str).str.strip()
    out["label_raw"] = df[spec["label_col"]]
    out["source"] = source_name

    if spec["label_map"]:
        out["label"] = out["label_raw"].map(spec["label_map"])
    else:
        out["label"] = out["label_raw"].astype(str)

    # Drop empty text
    out = out[out["text"].str.len() > 10].reset_index(drop=True)
    return out[["text", "label", "source"]]


def main():
    print("Week 1 — Kaggle Dataset Validation")
    print("=" * 55)

    frames = []
    for name, spec in DATASETS.items():
        df_raw = load_and_validate(name, spec)
        if df_raw is not None:
            df_std = standardise(df_raw, spec, source_name=name)
            frames.append(df_std)
            print(f"\nStandardised shape: {df_std.shape}")
            print(df_std.head(3))

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        out_path = os.path.join(PROCESSED_DIR, "kaggle_combined_raw.csv")
        combined.to_csv(out_path, index=False)
        print(f"\n[SAVED] Combined dataset → {out_path}")
        print(f"Total rows: {len(combined)}")
        print(f"Label counts:\n{combined['label'].value_counts()}")
    else:
        print("\n[INFO] No datasets loaded yet. Download CSVs from Kaggle first.")


if __name__ == "__main__":
    main()
