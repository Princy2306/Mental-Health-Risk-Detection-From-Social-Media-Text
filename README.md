# Mental Health Risk Detection from Social Media Text

An NLP-based machine learning system that detects potential mental health distress from social media text by classifying posts into **Low**, **Medium**, or **High** risk categories. The project combines publicly available datasets with an original survey dataset to build an interpretable and reliable prediction model.

---

## Project Overview

Mental health concerns are increasingly reflected in online conversations. Early identification of distress signals can help provide timely support and intervention.

This project develops an end-to-end NLP pipeline that analyzes textual data from social media and predicts the level of mental health risk using traditional machine learning and transformer-based language models.

---

## Features

- Detects mental health risk from textual posts
- Three-class classification:
  - Low Risk
  - Medium Risk
  - High Risk
- Uses both public datasets and self-collected survey data
- Text preprocessing and feature engineering
- Fine-tuned transformer models for improved accuracy
- Model explainability using SHAP
- Interactive prediction interface (planned)
- REST API deployment (planned)

---

## Dataset

### Public Dataset

- CLPsych Mental Health Dataset

### Custom Dataset

A Google Form survey collecting anonymous responses from **200–400 participants** over two weeks.

The survey includes:

- Emotion diary entries
- Sleep duration
- Stress levels
- Mood indicators
- Self-reported mental well-being

---

## Tech Stack

### Languages

- Python

### Libraries

- Pandas
- NumPy
- Scikit-learn
- Hugging Face Transformers
- PyTorch
- SHAP
- Matplotlib
- Seaborn

### NLP Techniques

- Text Cleaning
- Tokenization
- TF-IDF Vectorization
- BERT Embeddings
- Feature Engineering

---

## Machine Learning Pipeline

1. Data Collection
2. Data Cleaning
3. Text Preprocessing
4. Feature Extraction
5. Model Training
6. Hyperparameter Tuning
7. Model Evaluation
8. Explainability
9. Deployment

---

## Models Used

- Logistic Regression (Baseline)
- DistilBERT (Fine-tuned)
- Ensemble Model

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC-AUC

---

## Explainable AI

To improve transparency, SHAP (SHapley Additive Explanations) is used to understand which words and features contribute most to the model's predictions.

---

## Project Structure

```
Mental-Health-Risk-Detection/
│
├── data/
├── notebooks/
├── models/
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py
│
├── app/
├── requirements.txt
├── README.md
└── screenshots/
```

---

## Future Improvements

- Flask/FastAPI deployment
- Web application
- Browser extension
- Real-time prediction
- Fairness analysis across demographic groups
- Multi-language support
- Mobile application

---

## Results

*(Update after training)*

| Model | Accuracy | F1 Score |
|--------|----------|-----------|
| Logistic Regression | XX% | XX |
| DistilBERT | XX% | XX |
| Ensemble | XX% | XX |

---

## Installation

```bash
git clone https://github.com/yourusername/Mental-Health-Risk-Detection.git
```

```bash
cd Mental-Health-Risk-Detection
```

```bash
pip install -r requirements.txt
```

Run

```bash
python app.py
```

---

## Author

**Princy Nimmagadda**

B.Tech Student, IIT Jodhpur

Interested in Machine Learning, NLP, Data Science, and Software Engineering.
