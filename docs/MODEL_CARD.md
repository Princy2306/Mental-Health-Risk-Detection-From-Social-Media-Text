# Model Card — Mental Health Signal Detection

> Following Google's Model Card framework (Mitchell et al., 2019).  
> Last updated: [fill in date]

---

## Model Details

| Field | Value |
|-------|-------|
| Model type | Logistic Regression + DistilBERT (fine-tuned) |
| Input | Raw text (English) |
| Output | 3-class label: `low` / `moderate` / `high` risk |
| Label basis | PHQ-2 screening instrument (Kroenke et al., 2003) |
| Training data | Kaggle depression/mental health corpus + self-collected Google Form survey |
| Framework | scikit-learn 1.4 / HuggingFace Transformers 4.41 |
| Developer | [Your name], [College], [Year] |
| Version | 1.0 (Week 4 checkpoint) |

---

## Intended Use

**Primary intended use:**  
Academic research project exploring NLP-based mental health signal detection in student populations.

**Intended users:**  
Researchers and students studying NLP applications in mental health.

**Out-of-scope uses (do not use for):**
- Clinical diagnosis or treatment decisions
- Real-time monitoring of individuals without informed consent
- Any deployment without human clinician oversight
- Non-English text (model was not tested on multilingual data)
- Populations significantly different from college students (age 18–25)

---

## Training Data

### Sources
| Source | Size | Label method |
|--------|------|--------------|
| Kaggle: CLPsych / Depression Reddit | ~800 samples | Subreddit-based annotation |
| Google Form survey (college students) | ~300 samples | PHQ-2 self-report |

### Label distribution (training set)
| Class | Count | % |
|-------|-------|---|
| low | [fill] | [fill] |
| moderate | [fill] | [fill] |
| high | [fill] | [fill] |

### Data collection ethics
- All survey respondents provided informed consent
- No personally identifiable information was collected or stored
- Survey form included mental health resource signposting
- See `docs/ethics_note.md` for full ethics documentation

---

## Evaluation Results

### Classical baselines (TF-IDF + linguistic features)

| Model | F1-macro | CV F1 (5-fold) | Inference |
|-------|----------|----------------|-----------|
| Logistic Regression | [fill] | [fill] ± [fill] | [fill] ms |
| Linear SVM | [fill] | [fill] ± [fill] | [fill] ms |
| Random Forest | [fill] | [fill] ± [fill] | [fill] ms |

### DistilBERT fine-tuned

| Metric | Value |
|--------|-------|
| F1-macro (test) | [fill Week 4] |
| F1-weighted | [fill] |
| Inference time | [fill] ms |
| Model size | ~255 MB |

### Fairness audit results

| Demographic | Group | F1-macro | Recall (high-risk) | DPD |
|-------------|-------|----------|---------------------|-----|
| Gender | Male | [fill] | [fill] | [fill] |
| Gender | Female | [fill] | [fill] | — |
| Gender | Non-binary | [fill] | [fill] | — |

**Demographic Parity Difference (DPD):** [fill]  
**Interpretation:** [fill — e.g. "DPD of 0.07 on recall_high_risk between Male and Female groups. Flagged for investigation. Likely caused by dataset imbalance."]

---

## Key Findings from SHAP Analysis

**Most important tokens for `high` risk class:**
- [fill from notebook cell output]
- [fill]
- [fill]

**Most important linguistic features:**
- `negation_rate` — [fill observation]
- `fp_singular_rate` — [fill observation]
- `negative_affect` — [fill observation]

**Main error types (from error analysis):**
1. [fill — e.g. Sarcasm: "I'm absolutely fine" predicted as low, true label high]
2. [fill]
3. [fill]

---

## Limitations

1. **Dataset size**: ~800–1100 samples is small for deep learning. DistilBERT results may not generalise to larger diverse populations.

2. **Label reliability**: PHQ-2 is a screening tool, not a clinical instrument. Self-reported labels carry uncertainty and social desirability bias.

3. **Language**: English only. Mental health expression varies significantly across languages and cultures.

4. **Population**: Predominantly urban Indian college students aged 18–25. Generalisability to other populations is unknown.

5. **Temporal**: Cross-sectional data. Mental health is dynamic — a single text snapshot is a weak signal.

6. **Class imbalance**: The `high` risk class is underrepresented. Despite SMOTE/class weighting, model recall on this class may be lower in real deployment.

7. **Demographic gaps**: SHAP analysis shows [fill in finding]. The model should not be used for groups where DPD > 0.05 without mitigation.

---

## Ethical Considerations

- **False negatives are more harmful than false positives** in this domain. A model that misses a high-risk individual is worse than one that over-flags. Threshold tuning (lowering decision threshold for `high` class) is recommended before any deployment.
- **Stigma**: Any deployment must avoid stigmatising labelling. The model output should never be presented to individuals as a diagnosis.
- **Consent**: Any real-world text analysis application would require explicit informed consent and data minimisation practices.
- **Human oversight**: This model must never be deployed without a qualified human (counsellor, clinician) in the decision loop.

---

## Citation

```
[Your Name] (2025). Mental Health Signal Detection in Student Text: 
An NLP Pipeline with SHAP Explainability and Fairness Audit.
GitHub: https://github.com/YOUR_USERNAME/mental-health-nlp
```

---

*This Model Card was produced following Mitchell et al. (2019), "Model Cards for Model Reporting," FAccT.*
