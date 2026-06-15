"""
evaluator.py
Full classification evaluation: accuracy, F1, AUC, ROC curve, confusion matrix, PR curve.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
    confusion_matrix, roc_curve, precision_recall_curve, classification_report,
)


def clf_metrics(y_true, y_pred, y_prob=None) -> dict:
    m = {
        "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
    }
    if y_prob is not None:
        m["AUC-ROC"] = round(roc_auc_score(y_true, y_prob), 4)
    return m


def compare_models(fitted: dict, X_test, y_test) -> pd.DataFrame:
    rows = []
    for name, pipe in fitted.items():
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None
        m = clf_metrics(y_test, y_pred, y_prob)
        m["Model"] = name
        rows.append(m)
    return pd.DataFrame(rows).set_index("Model").sort_values("AUC-ROC", ascending=False)


def plot_roc_curves(fitted: dict, X_test, y_test, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, pipe in fitted.items():
        if hasattr(pipe, "predict_proba"):
            y_prob = pipe.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)
            ax.plot(fpr, tpr, lw=1.5, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves"); ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig


def plot_confusion(pipe, X_test, y_test, model_name="Model", save_path=None):
    y_pred = pipe.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"])
    ax.set_title(f"{model_name} – Confusion Matrix")
    ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig


def plot_pr_curves(fitted: dict, X_test, y_test, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, pipe in fitted.items():
        if hasattr(pipe, "predict_proba"):
            y_prob = pipe.predict_proba(X_test)[:, 1]
            prec, rec, _ = precision_recall_curve(y_test, y_prob)
            ax.plot(rec, prec, lw=1.5, label=name)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves"); ax.legend(fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig