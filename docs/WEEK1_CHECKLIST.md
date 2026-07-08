# Week 1 — Daily Checklist

Complete in order. Check off each item as you go.

---

## Day 1 — Environment + repo (1.5 hrs)

- [ ] Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) if not already installed
- [ ] Run in terminal:
  ```bash
  conda env create -f environment.yml
  conda activate mh-nlp
  python -m ipykernel install --user --name mh-nlp --display-name "mh-nlp"
  ```
- [ ] Create GitHub repo named `mental-health-nlp` (public, no template)
- [ ] Run `bash setup_git.sh` and follow the printed instructions
- [ ] Verify repo is live on GitHub with README visible

**Today's commit message**: `feat: initial project structure`

---

## Day 2 — Google Form design (1.5 hrs)

- [ ] Open Google Forms → New blank form
- [ ] Title: "Student Wellbeing & Academic Life Survey"
- [ ] Add consent checkbox as **first required question**:
  > "I understand this survey is anonymous, voluntary, and used only for academic research. I can stop at any time."  
  > Options: ✅ I agree (required)
- [ ] Add all sections from the question guide (see conversation above)
- [ ] Set PHQ-2 section title to "Daily wellbeing check-in" (not "depression screening")
- [ ] Add footer text on last page:
  > "If you're struggling, please reach out to your college counsellor or iCall: 9152987821"
- [ ] Preview form — complete it yourself as a test response
- [ ] Enable "Collect email addresses: OFF", "Limit to 1 response: OFF"
- [ ] Share form link to at least 3 WhatsApp groups / notice boards

**Today's commit message**: `docs: add ethics note, survey design doc`

---

## Day 3 — Kaggle data download (1 hr)

- [ ] Create a [Kaggle account](https://www.kaggle.com) if needed
- [ ] Download Dataset 1:
  - Go to: https://www.kaggle.com/datasets/infamouscoder/depression-reddit-cleaned
  - Download → place in `data/raw/depression_reddit.csv`
- [ ] Download Dataset 2:
  - Go to: https://www.kaggle.com/datasets/reihanenamdari/mental-health-corpus
  - Download → place in `data/raw/mental_health_corpus.csv`
- [ ] Run: `python data/download_kaggle_data.py`
- [ ] Verify output: `data/processed/kaggle_combined_raw.csv` exists

**Today's commit message**: `data: add kaggle download/validation script`

---

## Day 4 — EDA notebook (2 hrs)

- [ ] Run: `jupyter notebook`
- [ ] Open `notebooks/01_eda.ipynb`
- [ ] Run all cells in order
- [ ] Fill in **OBSERVATION** comments under each plot (required — not optional)
- [ ] Save all plots to `docs/` (the notebook does this automatically)
- [ ] Write Week 1 Summary table in the last cell

**Today's commit message**: `data: initial EDA on kaggle dataset`

---

## Day 5 — Column mapping + survey processor (1 hr)

- [ ] Open `data/process_survey.py`
- [ ] Update `COLUMN_MAP` dictionary:
  - Export your Google Form responses so far (even if < 20 responses)
  - Open the CSV, copy exact column headers
  - Paste into COLUMN_MAP on the left side
- [ ] Run: `python data/process_survey.py`
- [ ] Fix any "[WARNING] column not found" errors by updating the map
- [ ] Verify anonymised output looks correct (no names/emails)

**Today's commit message**: `data: add survey response processor script`

---

## Day 6–7 — Buffer + Week 1 review (1 hr)

- [ ] Check survey response count (target: 50+ by end of week 1)
- [ ] Send reminder to 2 more groups if < 30 responses
- [ ] Review GitHub commit history — should have 5+ commits
- [ ] Read `docs/ethics_note.md` — be ready to explain it in interviews
- [ ] Write 3 bullet points in a notes doc: "What I learned from the data this week"

---

## End-of-week targets

| Target | Status |
|--------|--------|
| GitHub repo live with README | ☐ |
| Google Form collecting responses | ☐ |
| Kaggle CSVs downloaded + validated | ☐ |
| EDA notebook complete with observations | ☐ |
| Ethics note written | ☐ |
| 5+ Git commits | ☐ |
