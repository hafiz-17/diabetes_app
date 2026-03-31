# 🏥 DiabetesAI – Flask Prediction System

AI-powered Diabetes Risk Assessment web application built with Flask + Gradient Boosting ML.

---

## 📁 Project Structure

```
diabetes_app/
├── app.py                          # Flask web application
├── train_model.py                  # ML training script
├── requirements.txt
├── diabetes_prediction_dataset.csv # Dataset (100K rows)
├── models/
│   ├── diabetes_model.pkl          # Trained GBM model
│   ├── scaler.pkl
│   ├── le_gender.pkl
│   ├── le_smoking.pkl
│   └── features.pkl
├── templates/
│   ├── index.html                  # Prediction page
│   └── analytics.html              # Model analytics dashboard
└── static/
    └── images/
        ├── learning_curves.png
        ├── confusion_matrix.png
        ├── roc_curve.png
        ├── pr_curve.png
        ├── feature_importance.png
        ├── class_distribution.png
        └── cv_scores.png
```

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model (generates model + graphs)
```bash
python train_model.py
```

### 3. Run the web app
```bash
python app.py
```

### 4. Open in browser
```
http://localhost:5000
```

---

## 🧠 Model Performance

| Metric | Score |
|---|---|
| Test Accuracy | 97.17% |
| ROC-AUC | 0.9790 |
| PR-AUC | 0.8838 |
| CV Accuracy (5-fold) | 97.18% ± 0.06% |

**Algorithm:** Gradient Boosting Classifier  
**Training Data:** 100,000 patient records  
**Features:** Blood Glucose, HbA1c, BMI, Age, Hypertension, Heart Disease, Smoking, Gender

---

## 🌟 Features

- **Risk Level Classification** – Low / Medium / High risk with probability score
- **Feature Importance Visualization** – Per-prediction factor analysis
- **PDF Health Report** – Downloadable medical-style PDF
- **Model Analytics Dashboard** – Learning curves, ROC, PR, Confusion Matrix, CV scores
- **Personalised Health Advice** – Based on risk level and input values

---

## 🔒 Disclaimer

This tool is for **educational and informational purposes only**.  
It does not constitute medical advice. Always consult a qualified physician.
