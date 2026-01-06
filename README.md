# Diabetes Prediction System

A machine learning application to predict diabetes risk based on health metrics using the Pima Indians Diabetes Dataset.

## Features

- 🤖 Multiple ML models (Logistic Regression, Random Forest, SVM)
- 📊 Comprehensive model evaluation and comparison
- 🌐 Beautiful Streamlit web interface
- 📈 Real-time predictions with probability scores

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Train the Model

```bash
python src/train_model.py
```

This will:
- Download the Pima Indians Diabetes Dataset
- Preprocess the data
- Train multiple ML models
- Save the best performing model

### 2. Run the Web Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Project Structure

```
diabetes_prediction/
├── app.py                  # Streamlit web application
├── requirements.txt        # Python dependencies
├── src/
│   ├── data_loader.py     # Data loading and preprocessing
│   └── train_model.py     # Model training pipeline
├── data/                   # Dataset storage
└── models/                 # Trained models
```

## Dataset

The Pima Indians Diabetes Dataset contains 768 samples with 8 features:
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

## Model Performance

The system trains and compares multiple models, selecting the best one based on ROC-AUC score.
