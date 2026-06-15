"""
data_loader.py
Loads the Pima Indians Diabetes / Heart Disease dataset.
Falls back to sklearn's breast cancer dataset + synthetic features.
"""
import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer


def load_raw(csv_path: str = None) -> pd.DataFrame:
    if csv_path and os.path.exists(csv_path):
        return pd.read_csv(csv_path)

    print("[data_loader] Using Breast Cancer dataset + synthetic patient features.")
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    df.rename(columns={"target": "Disease"}, inplace=True)
    # Disease=1 means benign in sklearn; flip so 1=malignant (disease present)
    df["Disease"] = 1 - df["Disease"]

    rng = np.random.default_rng(42)
    n = len(df)
    df["Age"]           = rng.integers(25, 80, size=n)
    df["Gender"]        = rng.choice(["M", "F"], size=n)
    df["Smoker"]        = rng.choice([0, 1], size=n, p=[0.7, 0.3])
    df["BloodPressure"] = rng.normal(120, 20, size=n).round(1).clip(60, 200)
    df["Cholesterol"]   = rng.normal(200, 40, size=n).round(1).clip(100, 350)
    df["InsuranceType"] = rng.choice(["Private", "Public", "None"], size=n, p=[0.5, 0.35, 0.15])

    # Inject ~3% missing values
    for col in ["Age", "BloodPressure", "Cholesterol", "InsuranceType"]:
        mask = rng.random(n) < 0.03
        df.loc[mask, col] = np.nan

    return df


def class_balance(df: pd.DataFrame, target: str = "Disease") -> dict:
    vc = df[target].value_counts()
    return {"counts": vc.to_dict(), "ratio": round(vc.max() / vc.min(), 2)}