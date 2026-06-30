"""
Fraud Detection Analytics
---------------------------
Analyzes transaction data and applies an Isolation Forest model to flag
potentially fraudulent transactions, then evaluates performance against
the (simulated) ground truth labels.

Outputs:
  - Console summary statistics & model evaluation
  - PNG charts saved to output/
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

DATA_PATH = "data/transactions.csv"
OUTPUT_DIR = "output"

plt.style.use("seaborn-v0_8-darkgrid")

FEATURES = ["amount", "distance_from_home_km", "transactions_last_24h"]


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    return df


def summary_stats(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("FRAUD DETECTION ANALYTICS — SUMMARY")
    print("=" * 60)
    print(f"Total transactions   : {len(df)}")
    print(f"Flagged as fraud     : {df['is_fraud'].sum()} "
          f"({df['is_fraud'].mean()*100:.2f}%)")
    print(f"Total amount         : Rs. {df['amount'].sum():,.2f}")
    print(f"Avg fraud amount     : Rs. {df[df.is_fraud==1]['amount'].mean():,.2f}")
    print(f"Avg legit amount     : Rs. {df[df.is_fraud==0]['amount'].mean():,.2f}")
    print()
    print("Fraud rate by merchant category:")
    print((df.groupby("merchant_category")["is_fraud"].mean() * 100)
          .round(2).sort_values(ascending=False).astype(str) + "%")
    print("=" * 60)


def train_isolation_forest(df: pd.DataFrame):
    """Unsupervised anomaly detection — does not use is_fraud labels for training,
    only for evaluating how well anomalies line up with real fraud."""
    X = df[FEATURES]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    contamination = max(df["is_fraud"].mean(), 0.01)
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
    )
    model.fit(X_scaled)

    # -1 = anomaly (predicted fraud), 1 = normal
    raw_pred = model.predict(X_scaled)
    df["predicted_fraud"] = (raw_pred == -1).astype(int)
    df["anomaly_score"] = model.decision_function(X_scaled)
    return df, model


def evaluate_model(df: pd.DataFrame) -> None:
    print("\nMODEL EVALUATION (Isolation Forest vs. ground truth)")
    print("-" * 60)
    print(classification_report(df["is_fraud"], df["predicted_fraud"],
                                 target_names=["Legitimate", "Fraud"]))
    cm = confusion_matrix(df["is_fraud"], df["predicted_fraud"])
    print("Confusion Matrix [rows=actual, cols=predicted]:")
    print(cm)
    print("-" * 60)
    return cm


def plot_confusion_matrix(cm) -> None:
    plt.figure(figsize=(5, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["Legit", "Fraud"])
    plt.yticks([0, 1], ["Legit", "Fraud"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                      color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=150)
    plt.close()


def plot_amount_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(df[df.is_fraud == 0]["amount"], bins=50, alpha=0.6, label="Legitimate", color="#4C72B0")
    plt.hist(df[df.is_fraud == 1]["amount"], bins=50, alpha=0.6, label="Fraud", color="#C44E52")
    plt.title("Transaction Amount Distribution: Legit vs Fraud")
    plt.xlabel("Amount (Rs. )")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/amount_distribution.png", dpi=150)
    plt.close()


def plot_fraud_by_hour(df: pd.DataFrame) -> None:
    data = df[df.is_fraud == 1].groupby("hour").size().reindex(range(24), fill_value=0)
    plt.figure(figsize=(9, 5))
    plt.bar(data.index, data.values, color="#C44E52")
    plt.title("Fraudulent Transactions by Hour of Day")
    plt.xlabel("Hour")
    plt.ylabel("Number of Fraud Cases")
    plt.xticks(range(24))
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fraud_by_hour.png", dpi=150)
    plt.close()


def plot_anomaly_scores(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    plt.scatter(df[df.is_fraud == 0]["amount"], df[df.is_fraud == 0]["anomaly_score"],
                alpha=0.4, s=10, label="Legitimate", color="#4C72B0")
    plt.scatter(df[df.is_fraud == 1]["amount"], df[df.is_fraud == 1]["anomaly_score"],
                alpha=0.7, s=15, label="Fraud", color="#C44E52")
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.title("Anomaly Score vs Transaction Amount")
    plt.xlabel("Amount (Rs. )")
    plt.ylabel("Isolation Forest Anomaly Score (lower = more anomalous)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/anomaly_scores.png", dpi=150)
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_data(DATA_PATH)
    summary_stats(df)
    df, model = train_isolation_forest(df)
    cm = evaluate_model(df)
    plot_confusion_matrix(cm)
    plot_amount_distribution(df)
    plot_fraud_by_hour(df)
    plot_anomaly_scores(df)
    df.to_csv(f"{OUTPUT_DIR}/transactions_with_predictions.csv", index=False)
    print(f"\nCharts and predictions saved to '{OUTPUT_DIR}/' directory.")


if __name__ == "__main__":
    main()
