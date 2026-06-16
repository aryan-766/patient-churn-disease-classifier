# 🩺 Patient Churn & Disease Risk Classification System

> End-to-End Machine Learning Classification Pipeline with Explainable AI, Hyperparameter Optimization, Ensemble Learning, and Interactive Streamlit Deployment.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-green)
![SHAP](https://img.shields.io/badge/SHAP-ExplainableAI-red)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b)

---

## 🚀 Overview

Healthcare organizations lose significant revenue and patient engagement when high-risk patients discontinue treatment or fail to return for follow-up care.

This project leverages Machine Learning and Explainable AI to:

✅ Predict patient churn risk

✅ Classify disease likelihood

✅ Identify high-risk patient segments

✅ Explain model decisions using SHAP

✅ Deploy predictions through an interactive Streamlit dashboard

---

## 🎯 Business Problem

Healthcare providers need proactive systems that can identify:

* Patients likely to discontinue treatment
* High-risk disease cases
* Key drivers influencing predictions
* Opportunities for early intervention

This solution helps healthcare teams make data-driven decisions and improve patient outcomes.

---

## ✨ Key Features

### 📊 Exploratory Data Analysis

* Missing value analysis
* Distribution visualizations
* Correlation heatmaps
* Class imbalance detection
* Feature relationship analysis

### 🧹 Data Preprocessing

* Missing value handling
* Feature encoding
* Scaling and normalization
* Outlier treatment
* Class balancing

### 🤖 Machine Learning Models

* Logistic Regression
* Random Forest
* XGBoost Classifier
* Voting Ensemble Classifier

### ⚙️ Hyperparameter Optimization

* Randomized Search CV
* Optuna Optimization
* Cross Validation
* Automated parameter tuning

### 📈 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* PR-AUC
* Confusion Matrix

### 🔍 Explainable AI

* SHAP Feature Importance
* Global Interpretability
* Local Prediction Explanations
* Model Transparency

### 🌐 Deployment

* Interactive Streamlit Dashboard
* Real-time Patient Prediction
* Risk Score Visualization
* Explainability Dashboard

---

## 🛠️ Tech Stack

| Category          | Tools                          |
| ----------------- | ------------------------------ |
| Language          | Python                         |
| Data Processing   | Pandas, NumPy                  |
| Visualization     | Matplotlib, Seaborn, Plotly    |
| Machine Learning  | Scikit-Learn                   |
| Ensemble Learning | Random Forest, Voting Ensemble |
| Boosting          | XGBoost                        |
| Optimization      | Optuna                         |
| Explainability    | SHAP                           |
| Deployment        | Streamlit                      |

---

## 📂 Project Structure

```text
patient_classifier/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_model.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── trainer.py
│   ├── evaluator.py
│   ├── tuner.py
│   └── explainer.py
│
├── app.py
├── requirements.txt
└── README.md
```

---

## 🔄 Machine Learning Workflow

```text
Raw Data
    ↓
EDA
    ↓
Preprocessing
    ↓
Feature Engineering
    ↓
Train/Test Split
    ↓
Cross Validation
    ↓
Model Training
    ↓
Hyperparameter Tuning
    ↓
Evaluation
    ↓
SHAP Explainability
    ↓
Streamlit Deployment
```

---

## 📊 Models Compared

| Model               | Purpose                    |
| ------------------- | -------------------------- |
| Logistic Regression | Baseline Linear Classifier |
| Random Forest       | Bagging Ensemble           |
| XGBoost             | Gradient Boosting          |
| Voting Classifier   | Ensemble Aggregation       |

---

## 🔬 Explainable AI with SHAP

The project integrates SHAP (SHapley Additive Explanations) to provide:

* Global feature importance
* Local prediction explanations
* Feature impact visualization
* Trustworthy model decisions

This helps healthcare professionals understand why a patient is classified as high risk.

---

## 📈 Sample Outputs

* ROC Curve
* Precision Recall Curve
* Confusion Matrix
* SHAP Summary Plot
* Feature Importance Plot
* Patient Risk Dashboard

---

## ▶️ Installation

```bash
git clone https://github.com/yaryan-766/patient-classifier.git

cd patient-classifier

python -m venv .venv

.venv\Scripts\activate

pip install -r requirements.txt
```

---

## ▶️ Run Streamlit App

```bash
streamlit run app.py
```

---

## 🧪 Future Improvements

* Deep Learning Models
* LLM-based Medical Assistant
* Real-Time Monitoring
* Cloud Deployment
* Docker Support
* CI/CD Pipeline
* Patient Recommendation Engine

---

## 👨‍💻 Author

Aryan Jain

AI/ML Engineer

Focused on:

* Machine Learning
* Deep Learning
* Explainable AI
* MLOps
* AI Agents
* Production AI Systems

---

⭐ If you found this project useful, consider giving it a star.
