# Week 2 — Daily Checklist
## EDA Deep Dive + Feature Engineering (~12 hrs total)

---

## Day 1 — Load data + clean (2 hrs)

- [ ] Activate environment: `conda activate mh-nlp`
- [ ] Confirm `data/processed/kaggle_combined_raw.csv` exists (from Week 1)
- [ ] Run smoke test: `python features.py`
  - Should print 24 features with values for the sample text
- [ ] Open `notebooks/02_features.ipynb` in Jupyter
- [ ] Run cells 0 (imports) and 1 (load + clean)
- [ ] Check: any null values in text column? Any unexpected labels?

**Today's commit**: `data: validate kaggle dataset, add clean text column`

---

## Day 2 — Class distribution + text length (2 hrs)

- [ ] Run cells 2 and 3 in `02_features.ipynb`
- [ ] Check plots are saved to `docs/` folder
- [ ] Fill in `# OBSERVATION:` comment in cell 2:
  - What is the exact imbalance ratio?
  - Which class is the minority? By how much?
- [ ] Fill in `# OBSERVATION:` in cell 3:
  - Do high-risk texts tend to be shorter? (Research shows depressive posts are often brief)
  - What % of texts exceed 380 words (DistilBERT limit)?
- [ ] Check survey response count — send reminders if < 100 responses so far

**Today's commit**: `data: EDA class distribution and text length analysis`

---

## Day 3 — TF-IDF token analysis (2 hrs)

- [ ] Run cell 4 (Top TF-IDF tokens per class)
- [ ] Fill in `# OBSERVATION:`:
  - Name 3 tokens that appear ONLY in high-risk texts
  - Name 3 tokens shared across all classes (noise)
  - Note one "surprising" token and hypothesise why it appears
- [ ] Screenshot or export the token chart — use it in your LinkedIn post later
- [ ] Run cell 5 (linguistic feature extraction)
- [ ] Verify: `feat_df.shape` shows 24 feature columns + label

**Today's commit**: `feat: implement and test 24 linguistic features`

---

## Day 4 — Correlation analysis + SMOTE (2.5 hrs)

- [ ] Run cells 6, 7, 8, 9
- [ ] Fill in `# OBSERVATION:` in cell 7:
  - Which 3 features have the highest correlation with the label?
  - Are any two features highly correlated with EACH OTHER (>0.85)?
    If yes, note them — you may want to drop one in Week 3
- [ ] Fill in cell 8 observation (TF-IDF sparsity)
- [ ] Decision: Write down which imbalance strategy you'll try first in Week 3
  - Option A: `class_weight='balanced'` (fast, no data generation)
  - Option B: SMOTE (synthetic samples, better for severe imbalance)
  - Option C: Both — compare F1-macro scores
- [ ] Run cell 10 (save outputs)
- [ ] Verify files exist:
  - `data/processed/features_kaggle.csv`
  - `data/processed/tfidf_kaggle.npz`
  - `models/tfidf_vectorizer.pkl`
  - `models/label_encoder.pkl`

**Today's commit**: `feat: correlation analysis, SMOTE experiment, save feature matrix`

---

## Day 5 — Survey data merge (1.5 hrs)

- [ ] Export Google Form responses as CSV
  - Google Forms → Responses tab → ⋮ menu → Download responses (.csv)
  - Save as `data/raw/survey_responses.csv`
- [ ] Run: `python data/process_survey.py`
  - Fix any column name mismatches (update `COLUMN_MAP` in the script)
- [ ] Run cell 11 in `02_features.ipynb` (survey merge)
- [ ] Verify `data/processed/features_combined.csv` exists
- [ ] Check label distribution in combined dataset — is it more or less balanced than Kaggle alone?

**Today's commit**: `data: merge survey responses with kaggle features`

---

## Day 6–7 — Buffer + review (1.5 hrs)

- [ ] Fill in the Week 2 Summary table at the bottom of the notebook
- [ ] Practice your 90-second EDA story (template in the notebook)
- [ ] Review all 6 saved plots in `docs/` — are they clean enough for LinkedIn?
- [ ] Close survey form if you have 250+ responses
- [ ] Check GitHub — should have 10+ commits total across weeks 1–2

---

## End-of-week targets

| Deliverable | Done? |
|-------------|-------|
| `02_features.ipynb` complete with all observations filled | ☐ |
| `features_kaggle.csv` saved (24 ling features) | ☐ |
| `tfidf_kaggle.npz` saved | ☐ |
| `models/tfidf_vectorizer.pkl` saved | ☐ |
| `features_combined.csv` (with survey data) | ☐ |
| Top 5 predictive features identified + explained | ☐ |
| 90-second EDA interview story written | ☐ |

---

## Interview prep — fill these in now

**"What surprised you in the EDA?"**
> [Your answer — specific token or feature finding]

**"Which feature was most predictive and why?"**
> [Your answer — from the correlation analysis]

**"How did you handle class imbalance?"**
> [Your answer — class_weight vs SMOTE comparison]

**"What does the type-token ratio measure?"**
> It measures vocabulary diversity. A lower ratio means the person is using the same words repeatedly, which can be a signal of rumination or limited emotional vocabulary — both documented in depression research.
