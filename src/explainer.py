"""
explainer.py
SHAP-based model explanation for black-box classifiers.
"""
import numpy as np
import matplotlib.pyplot as plt

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[explainer] shap not installed. pip install shap")


def get_shap_values(pipe, X_train_processed, X_test_processed):
    """Compute SHAP values for the model inside the pipeline."""
    if not SHAP_AVAILABLE:
        return None, None
    model = pipe.named_steps["model"]
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        explainer = shap.KernelExplainer(model.predict_proba, X_train_processed[:50])
    shap_values = explainer.shap_values(X_test_processed)
    return explainer, shap_values


def shap_summary_plot(shap_values, X_processed, feature_names, save_path=None):
    if not SHAP_AVAILABLE or shap_values is None:
        return None
    vals = shap_values[1] if isinstance(shap_values, list) else shap_values
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.summary_plot(vals, X_processed, feature_names=feature_names, show=False)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return plt.gcf()