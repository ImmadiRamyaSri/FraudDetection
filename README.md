💳 Credit Card Fraud Detection System
PythonScikit-LearnXGBoostImbalanced-LearnFastAPIDockerAWS

An end-to-end Machine Learning pipeline for detecting fraudulent credit card transactions in extremely imbalanced financial data — from EDA through production deployment with a real-time REST API on AWS EC2.

## 🚀 Live API Deployment
This model is currently deployed live on an **AWS EC2** instance using **Docker** and **FastAPI**.

*   **Test the API (Swagger UI):** [http://65.0.105.131:8000/docs](http://65.0.105.131:8000/docs)
*   **Health Check Endpoint:** [http://65.0.105.131:8000/health](http://65.0.105.131:8000/health)

*(Note: Click "Try it out" on the `/predict` endpoint in the Swagger UI to send a sample transaction and see the AI's risk stratification in real-time).*

📌 Project Overview
In financial fraud detection, class imbalance is a severe challenge. Standard accuracy is misleading because a dummy classifier predicting 100% legitimate transactions would achieve 99.83% accuracy while missing 100% of fraud.

This project implements robust feature preprocessing, SMOTE (Synthetic Minority Over-sampling Technique) on the training set to prevent data leakage, compares a Logistic Regression baseline against an optimized XGBoost Classifier, and deploys the final model as a containerized FastAPI microservice on AWS EC2.

🎯 Key Highlights
Dataset: 284,807 European cardholder transactions (September 2013).
Imbalance Ratio: Only 492 fraudulent transactions (0.17%).
False Positive Reduction: Cut false alarms from 1,458 down to 35 (a 97.6% reduction) using XGBoost.
Top Metric: Achieved 0.98 ROC-AUC and an F1-Score of 0.79 for the fraud class.
Production API: Real-time inference via FastAPI with risk-level stratification, containerized with Docker and deployed on AWS EC2.
🔬 Workflow & Technical Architecture

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
       │
       ▼
Model Export (joblib serialization of XGBoost + StandardScaler)
       │
       ▼
FastAPI REST API (Real-time /predict + /health endpoints)
       │
       ▼
Docker Containerization & AWS EC2 Deployment
📊 Exploratory Data Analysis & Insights
Transaction Amount Behavior:
The majority of fraudulent transactions were concentrated in lower amounts (< $500). Fraudsters intentionally make smaller purchases to bypass automated bank threshold alarms.
Transaction Time Patterns:
Legitimate transactions follow a 24-hour cyclical rhythm with peaks during daytime and dips at night.
Fraudulent transactions are more evenly distributed across all hours, indicating activity during off-peak times to avoid real-time human verification.
🛠️ Data Preprocessing & Balancing
Feature Scaling: Features V1 to V28 are already anonymized and scaled via PCA. Amount and Time were transformed using StandardScaler to prevent high-magnitude features from dominating gradient updates.
Stratified Splitting: An 80/20 train-test split was executed with stratify=y to preserve the 0.17% fraud ratio across both subsets.
Data Leakage Prevention with SMOTE: SMOTE was fitted only on X_train to synthesize minority class samples via K-Nearest Neighbor interpolation. X_test remained untouched to represent real-world evaluation.
📈 Model Performance & Comparison
Evaluated on the held-out test set of 56,962 transactions (98 Fraud, 56,864 Legit):

Metric	Logistic Regression (Baseline)	XGBoost Classifier (Final)	Winner
ROC-AUC Score	0.97	0.98	XGBoost
Fraud Precision	0.06 (6%)	0.71 (71%)	XGBoost (🏆 +65%)
Fraud Recall	0.92 (92%)	0.88 (88%)	Logistic Regression
Fraud F1-Score	0.11	0.79	XGBoost (🏆 +0.68)
False Positives (Innocent Flagged)	1,458	35	XGBoost (97.6% Drop)
False Negatives (Fraud Missed)	8	12	Logistic Regression
💡 Business Impact Analysis
The Problem with Baseline: Logistic Regression caught 92% of fraud but flagged 1,458 innocent customers. In production, blocking 1,458 legitimate transactions creates extreme customer friction and operational burden.
The XGBoost Solution: XGBoost caught 88% of all fraud cases while slashing false alarms to just 35, providing an optimal balance between risk management and user experience.
🌐 REST API — Real-Time Inference
The trained XGBoost model is served via a FastAPI microservice with the following endpoints:

Endpoints
Method	Endpoint	Description
GET	/health	Service health check and model status
POST	/predict	Score a single transaction for fraud risk
Risk Level Stratification
Fraud Probability	Risk Level
< 0.30	🟢 LOW
0.30 – 0.70	🟡 MEDIUM
0.70 – 0.90	🟠 HIGH
≥ 0.90	🔴 CRITICAL
Sample Request
bash

curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 406.0,
    "V1": -2.3122, "V2": 1.9519, "V3": -1.6098, "V4": 3.9979,
    "V5": -0.5221, "V6": -1.4265, "V7": -2.5373, "V8": 1.3916,
    "V9": -2.7700, "V10": -2.7722, "V11": 3.2020, "V12": -2.8999,
    "V13": -0.5952, "V14": -4.2892, "V15": 0.3897, "V16": -1.1407,
    "V17": -2.8300, "V18": -0.0168, "V19": 0.4169, "V20": 0.1269,
    "V21": 0.5172, "V22": -0.0350, "V23": -0.4652, "V24": 0.3201,
    "V25": 0.0445, "V26": 0.1778, "V27": 0.2611, "V28": -0.1432,
    "Amount": 0.00
  }'
Sample Response
json

{
  "is_fraud": true,
  "fraud_probability": 0.9983,
  "risk_level": "CRITICAL",
  "decision_threshold": 0.5,
  "latency_ms": 29.75
}
📁 Repository Structure

├── api/
│   ├── __init__.py
│   ├── main.py              # FastAPI application with /predict and /health endpoints
│   └── schemas.py           # Pydantic request/response models
├── src/
│   ├── __init__.py
│   └── train.py             # Standalone training script — exports model artifacts
├── models/
│   ├── fraud_xgboost.joblib # Serialized trained XGBoost classifier
│   └── scaler.joblib        # Fitted StandardScaler for Amount & Time
├── data/
│   └── creditcard.csv       # Kaggle Credit Card Fraud Dataset (not in repo)
├── notebooks/
│   └── FraudDetection.ipynb # Complete EDA, Preprocessing, Modeling & Visualizations
├── Dockerfile               # Container configuration for deployment
├── .dockerignore             # Files excluded from Docker build context
├── .gitignore                # Files excluded from version control
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
🚀 Getting Started
1. Clone the Repository
bash

git clone https://github.com/ImmadiRamyaSri/CreditCard-Fraud-Detection.git
cd CreditCard-Fraud-Detection
2. Set Up Environment & Install Dependencies
bash

python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
pip install -r requirements.txt
3. Train & Export Model
bash

python src/train.py
This generates models/fraud_xgboost.joblib and models/scaler.joblib.

4. Run API Locally
bash

uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
Open http://localhost:8000/docs for interactive Swagger documentation.

5. Deploy with Docker
bash

docker build -t fraud-detection-api .
docker run -d -p 8000:8000 fraud-detection-api
6. Deploy on AWS EC2
bash

# SSH into your EC2 instance
ssh -i "fraud-key.pem" ubuntu@<YOUR-EC2-PUBLIC-IP>
# Install Docker
sudo apt-get update && sudo apt-get install -y docker.io
sudo systemctl start docker && sudo systemctl enable docker
sudo usermod -aG docker ubuntu
# Clone, build, and run
git clone https://github.com/ImmadiRamyaSri/CreditCard-Fraud-Detection.git
cd CreditCard-Fraud-Detection
docker build -t fraud-detection-api .
docker run -d -p 8000:8000 fraud-detection-api
Access the live API at http://<YOUR-EC2-PUBLIC-IP>:8000/docs


🔮 Future Enhancements
 Add batch prediction endpoint (/predict/batch) for high-throughput scoring
 Implement model versioning and A/B testing support
 Add Prometheus metrics and Grafana monitoring dashboard
 Set up CI/CD pipeline with GitHub Actions for automated deployment
👤 Author
Ramya Sri Immadi ML / AI Engineer

GitHub:https://github.com/ImmadiRamyaSri
LinkedIn: https://www.linkedin.com/in/ramyasri-immadi-a0ab9218a/