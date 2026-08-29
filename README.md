# 💳 Credit Card Fraud Detection System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-green.svg)](https://xgboost.readthedocs.io/)
[![Imbalanced-Learn](https://img.shields.io/badge/Technique-SMOTE-red.svg)](https://imbalanced-learn.org/)

An end-to-end Machine Learning pipeline designed to detect fraudulent credit card transactions in extremely imbalanced real-world financial data.

---

## 📌 Project Overview

In financial fraud detection, class imbalance is a severe challenge. Standard accuracy is misleading because a dummy classifier predicting 100% legitimate transactions would achieve **99.83% accuracy** while missing **100% of fraud**.

This project implements robust feature preprocessing, **SMOTE (Synthetic Minority Over-sampling Technique)** on the training set to prevent data leakage, and compares a **Logistic Regression** baseline against an optimized **XGBoost Classifier**.

### 🎯 Key Highlights
* **Dataset:** 284,807 European cardholder transactions (September 2013).
* **Imbalance Ratio:** Only **492 fraudulent transactions (0.17%)**.
* **False Positive Reduction:** Cut false alarms from **1,458 down to 35 (a 97.6% reduction)** using XGBoost.
* **Top Metric:** Achieved **0.98 ROC-AUC** and an **F1-Score of 0.79** for the fraud class.

---

## 🔬 Workflow & Technical Architecture

```
Raw Data (284k rows)
       │
       ▼
Exploratory Data Analysis (Class Imbalance, Amount & Time Distribution)
       │
       ▼
Feature Scaling (StandardScaler on 'Amount' & 'Time'; V1-V28 are PCA-scaled)
       │
       ▼
Stratified Train-Test Split (80% Train / 20% Test)
       │
       ▼
SMOTE Oversampling (Applied STRICTLY on Training Data only)
       │
       ▼
Model Training & Evaluation (Logistic Regression vs. XGBoost)
       │
       ▼
Performance Comparison (Confusion Matrix, Precision-Recall, ROC-AUC Curves)
```

---

## 📊 Exploratory Data Analysis & Insights

1. **Transaction Amount Behavior:**
   - The majority of fraudulent transactions were concentrated in lower amounts (< $500). Fraudsters intentionally make smaller purchases to bypass automated bank threshold alarms.
2. **Transaction Time Patterns:**
   - Legitimate transactions follow a 24-hour cyclical rhythm with peaks during daytime and dips at night.
   - Fraudulent transactions are more evenly distributed across all hours, indicating activity during off-peak times to avoid real-time human verification.

---

## 🛠️ Data Preprocessing & Balancing

* **Feature Scaling:** Features `V1` to `V28` are already anonymized and scaled via PCA. `Amount` and `Time` were transformed using `StandardScaler` to prevent high-magnitude features from dominating gradient updates.
* **Stratified Splitting:** An 80/20 train-test split was executed with `stratify=y` to preserve the 0.17% fraud ratio across both subsets.
* **Data Leakage Prevention with SMOTE:** SMOTE was fitted **only on `X_train`** to synthesize minority class samples via K-Nearest Neighbor interpolation. `X_test` remained untouched to represent real-world evaluation.

---

## 📈 Model Performance & Comparison

Evaluated on the held-out test set of **56,962 transactions (98 Fraud, 56,864 Legit)**:

| Metric | Logistic Regression (Baseline) | XGBoost Classifier (Final) | Winner |
| :--- | :---: | :---: | :---: |
| **ROC-AUC Score** | 0.97 | **0.98** | **XGBoost** |
| **Fraud Precision** | 0.06 (6%) | **0.71 (71%)** | **XGBoost (🏆 +65%)** |
| **Fraud Recall** | **0.92 (92%)** | 0.88 (88%) | **Logistic Regression** |
| **Fraud F1-Score** | 0.11 | **0.79** | **XGBoost (🏆 +0.68)** |
| **False Positives (Innocent Flagged)** | 1,458 | **35** | **XGBoost (97.6% Drop)** |
| **False Negatives (Fraud Missed)** | **8** | 12 | **Logistic Regression** |

### 💡 Business Impact Analysis
* **The Problem with Baseline:** Logistic Regression caught 92% of fraud but flagged **1,458 innocent customers**. In production, blocking 1,458 legitimate transactions creates extreme customer friction and operational burden.
* **The XGBoost Solution:** XGBoost caught **88% of all fraud cases** while slashing false alarms to just **35**, providing an optimal balance between risk management and user experience.

---

## 📁 Repository Structure

```
├── data/
│   └── creditcard.csv           # Kaggle Credit Card Fraud Dataset
├── notebooks/
│   └── FraudDetection.ipynb     # Complete EDA, Preprocessing, Modeling & Visualizations
├── images/                      # Generated Heatmaps and ROC-AUC curves
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/ImmadiRamyaSri/CreditCard-Fraud-Detection.git
cd CreditCard-Fraud-Detection
```

### 2. Set up virtual environment & install dependencies
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run Jupyter Notebook
```bash
jupyter notebook notebooks/FraudDetection.ipynb
```

---

## 🔮 Future Enhancements
- [ ] Export trained XGBoost model artifacts using `joblib`.
- [ ] Build a REST API with **FastAPI** for real-time transaction inference.
- [ ] Containerize application with **Docker**.
- [ ] Deploy inference endpoint on **AWS EC2**.

---

## 👤 Author
**Ramya Sri Immadi**  
*ML / AI Engineer*  
- GitHub: [@ImmadiRamyaSri](https://github.com/ImmadiRamyaSri)  
- LinkedIn: [Ramya Sri Immadi](https://www.linkedin.com/in/immadiramyasri/)
