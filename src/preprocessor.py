"""
preprocessor.py
Handles class imbalance (SMOTE), imputation, scaling, encoding.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def split_features_target(df: pd.DataFrame, target: str = "Disease"):
    return df.drop(columns=[target]), df[target].copy()


def build_preprocessor(num_cols: list, cat_cols: list) -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols),
    ], remainder="drop")


def get_feature_groups(df: pd.DataFrame, target: str = "Disease"):
    num_cols, cat_cols = [], []
    for col in df.drop(columns=[target]).columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            num_cols.append(col)
        else:
            cat_cols.append(col)
    return num_cols, cat_cols


def apply_smote(X_train, y_train, random_state: int = 42):
    """Oversample minority class with SMOTE (only on training data!)."""
    try:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=random_state, k_neighbors=5)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        print(f"[SMOTE] {len(y_train)} → {len(y_res)} samples")
        return X_res, y_res
    except ImportError:
        print("[SMOTE] imbalanced-learn not installed – skipping.")
        return X_train, y_train