# Data Ethics Note

**Project**: Mental Health Signal Detection in Social Text  
**Data collection date**: [Fill in your dates]  
**Collected by**: [Your name], [Your college]

---

## 1. Informed Consent

All survey participants were informed of the following before responding:

- The survey is **completely voluntary** and can be abandoned at any time.
- **No personally identifiable information** (name, email, student ID, phone) is collected.
- Responses are used **only for academic ML research**.
- Data will be **anonymised and aggregated** — individual responses will not be reported.
- The PHQ-2 screening questions are **not a clinical diagnosis tool**.

A consent checkbox was the first required question on the form. Responses without consent were excluded.

## 2. Data Minimisation

We collect only what is necessary:

| Collected | Reason | Not collected |
|-----------|--------|---------------|
| Age range (bucketed) | Demographic feature | Exact age / DOB |
| Gender | Fairness audit | Name |
| PHQ-2 score | Ground truth label | Contact info |
| Open text (feelings) | NLP corpus | Location |
| Lifestyle signals | Feature engineering | Social media handles |

## 3. Storage and Access

- Raw CSV exported from Google Forms is stored **locally only**, never pushed to GitHub (see `.gitignore`).
- Only the anonymised, processed version (no free-text linking possible) is used in notebooks.
- Data is deleted from Google Drive after the project is complete.

## 4. Mental Health Safeguarding

- The Google Form included a footer: *"If you are struggling, please reach out to your college counsellor or iCall (India): 9152987821."*
- The PHQ-2 section was titled "Daily wellbeing check-in" to avoid stigma-driven drop-off.
- This model is **not intended for clinical use**. It is a research prototype only.

## 5. Limitations and Risks

- The model reflects biases in the training data (predominantly college-aged, English-speaking students).
- PHQ-2 is a screening tool, not a diagnostic instrument — labels carry uncertainty.
- Deployment in any real-world mental health context would require clinical validation, regulatory review, and human oversight.

## 6. IRB / Institutional Review

This project was conducted as an academic exercise and does not constitute formal human subjects research under IRB jurisdiction. However, the data collection followed IRB-inspired principles of informed consent, data minimisation, and participant protection.

---

*This document was written to accompany the project and will be cited in the Model Card.*
