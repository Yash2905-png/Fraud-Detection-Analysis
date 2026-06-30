"""
Generates a synthetic credit-card-style transaction dataset for fraud
detection analysis. Run this first to create data/transactions.csv
"""
import pandas as pd
import numpy as np

np.random.seed(21)

N_NORMAL = 4800
N_FRAUD = 200  # ~4% fraud rate, similar to common public fraud datasets

MERCHANT_CATEGORIES = ["Grocery", "Electronics", "Travel", "Dining",
                        "Online Retail", "Fuel", "Entertainment", "Utilities"]

start_date = pd.Timestamp("2025-01-01")

def make_normal(n):
    return pd.DataFrame({
        "amount": np.round(np.random.gamma(2, 40, n), 2),
        "hour": np.random.normal(14, 4, n).clip(0, 23).astype(int),
        "merchant_category": np.random.choice(MERCHANT_CATEGORIES, n),
        "distance_from_home_km": np.round(np.abs(np.random.normal(5, 5, n)), 1),
        "transactions_last_24h": np.random.poisson(2, n),
        "is_fraud": 0,
    })

def make_fraud(n):
    return pd.DataFrame({
        # fraud tends to be larger amounts, odd hours, far from home, bursty
        "amount": np.round(np.random.gamma(3, 150, n), 2),
        "hour": np.random.choice(list(range(0, 6)) + list(range(22, 24)), n),
        "merchant_category": np.random.choice(MERCHANT_CATEGORIES, n,
                                               p=[0.05, 0.25, 0.2, 0.05, 0.3, 0.05, 0.05, 0.05]),
        "distance_from_home_km": np.round(np.abs(np.random.normal(80, 60, n)), 1),
        "transactions_last_24h": np.random.poisson(7, n),
        "is_fraud": 1,
    })

df = pd.concat([make_normal(N_NORMAL), make_fraud(N_FRAUD)], ignore_index=True)

df["transaction_id"] = ["TXN" + str(i).zfill(6) for i in range(1, len(df) + 1)]
df["date"] = start_date + pd.to_timedelta(np.random.randint(0, 90, len(df)), unit="D")
df["timestamp"] = df["date"] + pd.to_timedelta(df["hour"], unit="h")

df = df[["transaction_id", "timestamp", "amount", "merchant_category",
         "distance_from_home_km", "transactions_last_24h", "is_fraud"]]

# shuffle so fraud rows aren't all at the bottom
df = df.sample(frac=1, random_state=1).reset_index(drop=True)

df.to_csv("data/transactions.csv", index=False)
print(f"Generated {len(df)} transactions "
      f"({df['is_fraud'].sum()} fraudulent, {(df['is_fraud'].mean()*100):.1f}%) "
      f"-> data/transactions.csv")
