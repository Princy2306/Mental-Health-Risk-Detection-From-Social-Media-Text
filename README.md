# Mental Health Risk Detection from Social Media Text

An end-to-end NLP application that detects potential mental health distress from social media text using **TF-IDF** and **Logistic Regression**. The project includes text preprocessing, model training, explainability, a Flask REST API, and an interactive Streamlit web application.

## Features

- Binary mental health risk classification (Low / High)
- TF-IDF feature extraction
- Logistic Regression classifier
- Explainable predictions with important token highlighting
- Streamlit web interface
- Flask REST API
- Batch prediction support
- Automated testing with Pytest
- Docker-ready deployment

## Tech Stack

- Python
- Scikit-learn
- Pandas
- NumPy
- SHAP
- Flask
- Streamlit
- Pytest
- Docker

## Project Structure

```
├── app/
├── data/
├── docs/
├── models/
├── notebooks/
├── tests/
├── Dockerfile
├── requirements.txt
└── README.md
```

## API Endpoints

- `/health`
- `/model-info`
- `/predict`
- `/predict-batch`

## Installation

```bash
git clone https://github.com/Princy2306/Mental-Health-Risk-Detection-From-Social-Media-Text.git

cd Mental-Health-Risk-Detection-From-Social-Media-Text

pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

Run the Flask API:

```bash
python app/api.py
```
