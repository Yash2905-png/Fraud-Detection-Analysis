# Fraud Detection Analytics

A data analytics project that uses unsupervised machine learning
(Isolation Forest) to detect anomalous, potentially fraudulent
transactions in transaction data.

## 📌 Project Overview

This project simulates a realistic transaction dataset (with a ~4% fraud
rate, similar to real-world card fraud datasets) and applies an Isolation
Forest model to flag anomalies based on:
- Transaction amount
- Distance from cardholder's home
- Number of transactions in the last 24 hours
- Time of day

It then evaluates how well the unsupervised model's predictions align
with the actual fraud labels.

## 🛠️ Tech Stack
- Python 3.8+
- pandas — data manipulation
- scikit-learn — Isolation Forest model, scaling, evaluation metrics
- matplotlib — visualization

## 📂 Project Structure
```
fraud-detection-analytics/
├── generate_data.py     # Creates the synthetic transaction dataset
├── analyze.py             # Main analysis, model training & evaluation
├── requirements.txt
├── data/
│   └── transactions.csv   (generated)
└── output/
    ├── confusion_matrix.png
    ├── amount_distribution.png
    ├── fraud_by_hour.png
    ├── anomaly_scores.png
    └── transactions_with_predictions.csv
```

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd fraud-detection-analytics

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate the dataset
python generate_data.py

# 4. Run the analysis
python analyze.py
```

## 🧠 Methodology

1. **Data generation**: synthetic transactions are created with realistic
   fraud signals — larger amounts, odd hours (late night/early morning),
   greater distance from home, and higher transaction frequency.
2. **Feature scaling**: numeric features are standardized with `StandardScaler`.
3. **Model**: an `IsolationForest` is trained in an unsupervised manner
   (it never sees the `is_fraud` label during training) to isolate anomalies.
4. **Evaluation**: predictions are compared against the true labels using
   a classification report and confusion matrix to measure how well
   anomaly detection approximates actual fraud detection.

## 📊 Sample Insights
- Fraud rate broken down by merchant category
- Average transaction amount: fraud vs legitimate
- Confusion matrix showing detection accuracy
- Distribution of fraud cases by hour of day
- Anomaly score scatter plot showing model confidence

## 🔮 Possible Extensions
- Compare against a supervised model (Logistic Regression, Random Forest, XGBoost)
- Add more behavioral features (spending velocity, device fingerprint, merchant risk score)
- Build a real-time scoring API with FastAPI
- Apply SMOTE or other techniques to handle class imbalance for supervised models

## 👤 Author
Built as part of a CodTech IT Solutions internship task.

## 📄 License
This project is open source under the MIT License.
