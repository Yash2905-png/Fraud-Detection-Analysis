# 🛡️ Fraud Detection Analytics

## CodTech IT Solutions — Data Analytics Internship | Task 4

---

| **Field** | **Details** |
|-----------|-------------|
| **Intern Name** | Yash Gamare |
| **Intern ID** | CITS4255 |
| **Domain** | Data Science & Analytics |
| **Project** | Fraud Detection Analytics |
| **Level** | Advanced |

---

## 📖 Project Overview

This project focuses on detecting potentially fraudulent financial transactions using **Unsupervised Machine Learning**. It utilizes the **Isolation Forest** algorithm to identify anomalies in transaction data without requiring labeled data during training.

The project simulates a realistic transaction dataset with approximately **4% fraudulent transactions**, analyzes suspicious patterns, and evaluates how effectively the anomaly detection model identifies fraud.

---

## 🎯 Objectives

- Detect fraudulent transactions using Isolation Forest.
- Analyze transaction patterns and fraud indicators.
- Compare predicted anomalies with actual fraud labels.
- Visualize fraud trends and anomaly scores.
- Evaluate model performance using classification metrics.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Git & GitHub

---

## 📂 Project Structure

```text
Fraud-Detection-Analytics/
│
├── generate_data.py
├── analyze.py
├── requirements.txt
├── README.md
│
├── data/
│   └── transactions.csv
│
└── output/
    ├── confusion_matrix.png
    ├── amount_distribution.png
    ├── fraud_by_hour.png
    ├── anomaly_scores.png
    └── transactions_with_predictions.csv
```

---

## 🚀 Installation

```bash
git clone https://github.com/YOUR_USERNAME/Fraud-Detection-Analytics.git

cd Fraud-Detection-Analytics

pip install -r requirements.txt

python generate_data.py

python analyze.py
```

---

## 📊 Features

- 💳 Fraud Detection using Isolation Forest
- 📈 Transaction Pattern Analysis
- 📊 Confusion Matrix Visualization
- ⏰ Fraud Analysis by Time of Day
- 💰 Transaction Amount Distribution
- 🔥 Anomaly Score Visualization
- 📂 Synthetic Dataset Generation

---

## 🧠 Methodology

1. Generate a realistic synthetic transaction dataset.
2. Clean and preprocess transaction data.
3. Standardize numerical features using `StandardScaler`.
4. Train the **Isolation Forest** model for anomaly detection.
5. Predict fraudulent transactions.
6. Compare predictions with actual labels.
7. Visualize results and evaluate model performance.

---

## 📈 Output

The project generates the following visualizations:

- 📌 Confusion Matrix
- 📌 Transaction Amount Distribution
- 📌 Fraud by Hour Analysis
- 📌 Anomaly Score Scatter Plot
- 📌 Transactions with Predicted Fraud Labels

All generated files are automatically saved inside the **output/** folder.

---

## 📊 Sample Insights

- Fraud vs Legitimate Transaction Comparison
- Average Transaction Amount Analysis
- Fraud Distribution Across Different Hours
- Anomaly Detection Performance
- Model Evaluation using Classification Metrics

---

## 🔮 Future Enhancements

- 🤖 Compare with Supervised Machine Learning Models
- 📊 Random Forest & XGBoost Implementation
- 🌐 Real-Time Fraud Detection API using FastAPI
- 📈 Interactive Streamlit Dashboard
- 🧠 Behavioral Pattern Analysis
- 📱 Live Transaction Monitoring Dashboard

---

## 👨‍💻 Author

**Yash Gamare**

Computer Engineering Graduate

GitHub: https://github.com/Yash2905-png

---

## 📄 License

This project is licensed under the **MIT License**.

---

⭐ If you found this project useful, consider giving it a **Star ⭐** on GitHub!
