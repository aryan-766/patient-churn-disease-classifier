"""
trainer.py
Logistic Regression (gradient descent), RF, XGBoost, VotingClassifier.
All wrapped in sklearn Pipelines.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    BaggingClassifier,
    AdaBoostClassifier,
    StackingClassifier,
)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
import joblib
import warnings
warnings.filterwarnings("ignore")


CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# ── Individual pipelines ───────────────────────────────────────────────────────

def lr_pipeline(preprocessor):
    """Logistic Regression – trained via (saga) gradient descent."""
    return Pipeline([
        ("prep", preprocessor),
        ("model", LogisticRegression(
            solver="saga",       # gradient descent-based solver
            penalty="l2",
            C=1.0,
            max_iter=500,
            random_state=42,
        )),
    ])


def sgd_pipeline(preprocessor):
    """Pure stochastic gradient descent classifier."""
    return Pipeline([
        ("prep", preprocessor),
        ("model", SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            max_iter=200,
            random_state=42,
        )),
    ])


def rf_pipeline(preprocessor):
    return Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=200, max_depth=8,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )),
    ])


def gbm_pipeline(preprocessor):
    return Pipeline([
        ("prep", preprocessor),
        ("model", GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=4,
            subsample=0.8, random_state=42,
        )),
    ])


def xgb_pipeline(preprocessor):
    return Pipeline([
        ("prep", preprocessor),
        ("model", XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.7,
            use_label_encoder=False, eval_metric="logloss",
            scale_pos_weight=1, random_state=42, verbosity=0,
        )),
    ])


def adaboost_pipeline(preprocessor):
    return Pipeline([
        ("prep", preprocessor),
        ("model", AdaBoostClassifier(n_estimators=200, learning_rate=0.05, random_state=42)),
    ])


def bagging_pipeline(preprocessor):
    return Pipeline([
        ("prep", preprocessor),
        ("model", BaggingClassifier(
            estimator=RandomForestClassifier(n_estimators=50, random_state=42),
            n_estimators=10, random_state=42, n_jobs=-1,
        )),
    ])


# ── Ensemble: Voting ───────────────────────────────────────────────────────────

def voting_pipeline(preprocessor):
    return Pipeline([
        ("prep", preprocessor),
        ("model", VotingClassifier(
            estimators=[
                ("lr",  LogisticRegression(solver="saga", C=1.0, max_iter=500, random_state=42)),
                ("rf",  RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)),
                ("xgb", XGBClassifier(n_estimators=200, learning_rate=0.05, random_state=42, verbosity=0, eval_metric="logloss")),
            ],
            voting="soft",
            n_jobs=-1,
        )),
    ])


# ── Ensemble: Stacking ─────────────────────────────────────────────────────────

def stacking_pipeline(preprocessor):
    return Pipeline([
        ("prep", preprocessor),
        ("model", StackingClassifier(
            estimators=[
                ("rf",  RandomForestClassifier(n_estimators=150, class_weight="balanced", random_state=42, n_jobs=-1)),
                ("xgb", XGBClassifier(n_estimators=200, learning_rate=0.1, random_state=42, verbosity=0, eval_metric="logloss")),
                ("gbm", GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42)),
            ],
            final_estimator=LogisticRegression(C=1.0, max_iter=500),
            cv=5,
            passthrough=False,
            n_jobs=-1,
        )),
    ])


# ── Train helpers ──────────────────────────────────────────────────────────────

def train_all(X_train, y_train, preprocessor):
    models = {
        "LogisticReg (GD)": lr_pipeline(preprocessor),
        "SGD Classifier":   sgd_pipeline(preprocessor),
        "RandomForest":     rf_pipeline(preprocessor),
        "GBM":              gbm_pipeline(preprocessor),
        "XGBoost":          xgb_pipeline(preprocessor),
        "AdaBoost":         adaboost_pipeline(preprocessor),
        "Bagging":          bagging_pipeline(preprocessor),
        "Voting":           voting_pipeline(preprocessor),
        "Stacking":         stacking_pipeline(preprocessor),
    }
    fitted = {}
    for name, pipe in models.items():
        print(f"  Training {name}...", end=" ", flush=True)
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        print("done")
    return fitted


def cv_score(pipe, X, y, n_splits=5, scoring="roc_auc"):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=kf, scoring=scoring, n_jobs=-1)
    return scores.mean(), scores.std()