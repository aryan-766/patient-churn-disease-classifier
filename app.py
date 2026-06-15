"""
app.py  —  Patient Disease Classifier · Streamlit Frontend
Run:  streamlit run app.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from src.data_loader import load_raw, class_balance
from src.preprocessor import build_preprocessor, split_features_target, get_feature_groups
from src.trainer import train_all, cv_score
from src.evaluator import compare_models, plot_roc_curves, plot_confusion, plot_pr_curves

st.set_page_config(page_title="Patient Disease Classifier", page_icon="🏥", layout="wide")

st.markdown("""
<style>
    .stMetric label {font-size:0.75rem!important;color:#6c757d}
    .alert-box{padding:1rem;border-radius:10px;border:1px solid}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("🏥 Patient Classifier")
st.sidebar.markdown("---")
test_size   = st.sidebar.slider("Test split %", 10, 40, 20) / 100
n_cv_folds  = st.sidebar.slider("CV folds", 3, 10, 5)
use_smote   = st.sidebar.checkbox("Use SMOTE (imbalance)", True)
selected    = st.sidebar.multiselect(
    "Models to train",
    ["LogisticReg (GD)", "SGD Classifier", "RandomForest", "GBM",
     "XGBoost", "AdaBoost", "Bagging", "Voting", "Stacking"],
    default=["LogisticReg (GD)", "RandomForest", "XGBoost", "Voting", "Stacking"],
)
run_btn = st.sidebar.button("🚀  Train models", use_container_width=True)

# ── Main ───────────────────────────────────────────────────────────────────────
st.title("🏥 Patient Disease Classification Pipeline")
st.markdown("Gradient descent · XGBoost · Ensemble · SHAP · ROC · Confusion Matrix")
st.markdown("---")

tabs = st.tabs(["📊 EDA", "⚖️ Class Balance", "🤖 Training & CV", "📈 Evaluation", "🎛️ Predict"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 – EDA
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.header("Exploratory Data Analysis")
    df = load_raw()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Features", df.shape[1] - 1)
    c3.metric("Disease cases", int(df["Disease"].sum()))
    c4.metric("Missing %", f"{df.isnull().mean().mean()*100:.1f}%")

    st.subheader("Sample data")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Feature distributions")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    sel_feat = st.selectbox("Select feature", [c for c in num_cols if c != "Disease"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for cls, grp in df.groupby("Disease"):
        axes[0].hist(grp[sel_feat].dropna(), alpha=0.5, bins=30,
                     label=f"Disease={cls}", edgecolor="white")
    axes[0].set_title(f"{sel_feat} by Disease status"); axes[0].legend()
    df.boxplot(column=sel_feat, by="Disease", ax=axes[1])
    axes[1].set_title(f"{sel_feat} — Box plot by class")
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Pairplot (top correlated features)")
    corrs = df.select_dtypes(include=np.number).corr()["Disease"].abs().sort_values(ascending=False)
    top4  = corrs.index[1:5].tolist() + ["Disease"]
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    sns.heatmap(df[top4].corr(), annot=True, cmap="RdBu_r", center=0, ax=ax2)
    st.pyplot(fig2)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 – Class Balance
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.header("Class Balance & Imbalance Handling")
    df = load_raw()
    bal = class_balance(df)
    c1, c2 = st.columns(2)
    c1.metric("Class 0 (No disease)", bal["counts"].get(0, 0))
    c2.metric("Class 1 (Disease)",    bal["counts"].get(1, 0))
    st.metric("Imbalance ratio", f"{bal['ratio']}:1")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    vals = list(bal["counts"].values())
    axes[0].bar(["No Disease", "Disease"], vals, color=["steelblue","tomato"])
    axes[0].set_title("Class Distribution (before SMOTE)")
    if use_smote:
        st.info("✅  SMOTE will oversample the minority class on training data only.")
        axes[1].bar(["No Disease", "Disease"], [max(vals), max(vals)],
                    color=["steelblue","tomato"])
        axes[1].set_title("After SMOTE (approximate)")
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("""
**Why SMOTE?**
- SMOTE (Synthetic Minority Oversampling TEchnique) generates synthetic examples of the minority class
- Applied *only* to training data to avoid data leakage
- Prevents the model from always predicting the majority class
""")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 – Training & CV
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.header("Model Training & Stratified Cross Validation")

    if run_btn:
        df = load_raw()
        X, y = split_features_target(df)
        num_cols, cat_cols = get_feature_groups(df)
        prep = build_preprocessor(num_cols, cat_cols)
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42)

        if use_smote:
            from src.preprocessor import apply_smote
            X_tr_t = prep.fit_transform(X_train_raw, y_train)
            X_tr_t, y_tr_s = apply_smote(X_tr_t, y_train)
            # We need raw for the Pipeline — rebuild with fit-only
            prep2 = build_preprocessor(num_cols, cat_cols)
            prep2.fit(X_train_raw)
        else:
            prep2 = build_preprocessor(num_cols, cat_cols)
            prep2.fit(X_train_raw)

        with st.spinner("Training models…"):
            fitted = train_all(X_train_raw, y_train, prep2)
            fitted = {k: v for k, v in fitted.items() if k in selected}

        st.session_state.update({
            "clf_fitted": fitted,
            "clf_X_test": X_test_raw,
            "clf_y_test": y_test,
            "clf_X_train": X_train_raw,
            "clf_y_train": y_train,
            "clf_prep": prep2,
        })
        st.success("✅  Training complete!")

        st.subheader(f"Stratified {n_cv_folds}-Fold CV — AUC-ROC")
        rows = []
        for name, pipe in fitted.items():
            mu, std = cv_score(pipe, X_train_raw, y_train, n_splits=n_cv_folds, scoring="roc_auc")
            f1_mu, f1_std = cv_score(pipe, X_train_raw, y_train, n_splits=n_cv_folds, scoring="f1")
            rows.append({"Model": name, "AUC-ROC": round(mu, 4), "AUC std": round(std, 4),
                         "F1": round(f1_mu, 4)})
        cv_df = pd.DataFrame(rows).sort_values("AUC-ROC", ascending=False)
        st.dataframe(cv_df.set_index("Model"), use_container_width=True)

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.barh(cv_df["Model"], cv_df["AUC-ROC"], color="steelblue")
        ax.set_xlim(0.5, 1.0); ax.set_xlabel("Mean CV AUC-ROC")
        ax.set_title("Cross Validation AUC-ROC")
        st.pyplot(fig)
    else:
        st.info("👈  Configure and click Train models.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 – Evaluation
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.header("Test Set Evaluation")
    if "clf_fitted" in st.session_state:
        fitted  = st.session_state["clf_fitted"]
        X_test  = st.session_state["clf_X_test"]
        y_test  = st.session_state["clf_y_test"]

        metrics_df = compare_models(fitted, X_test, y_test)
        st.subheader("Metrics table")
        st.dataframe(metrics_df.style.highlight_max(axis=0, color="#d4f1e4"), use_container_width=True)

        st.subheader("ROC Curves")
        fig_roc = plot_roc_curves(fitted, X_test, y_test)
        st.pyplot(fig_roc)

        st.subheader("Precision-Recall Curves")
        fig_pr = plot_pr_curves(fitted, X_test, y_test)
        st.pyplot(fig_pr)

        st.subheader("Confusion Matrix")
        sel = st.selectbox("Select model", list(fitted.keys()))
        fig_cm = plot_confusion(fitted[sel], X_test, y_test, model_name=sel)
        st.pyplot(fig_cm)
    else:
        st.info("Run training first (Tab 3).")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 – Live Predict
# ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.header("🎛️  Live Patient Risk Prediction")
    if "clf_fitted" in st.session_state:
        fitted = st.session_state["clf_fitted"]
        model_choice = st.selectbox("Model", list(fitted.keys()))

        with st.form("clf_form"):
            c1, c2, c3 = st.columns(3)
            age     = c1.number_input("Age", 18, 90, 45)
            gender  = c2.selectbox("Gender", ["M", "F"])
            smoker  = c3.selectbox("Smoker", [0, 1], format_func=lambda x: "Yes" if x else "No")
            bp      = c1.number_input("Blood Pressure (mmHg)", 60, 200, 120)
            chol    = c2.number_input("Cholesterol (mg/dL)", 100, 350, 200)
            instype = c3.selectbox("Insurance", ["Private", "Public", "None"])
            submitted = st.form_submit_button("Predict risk →")

        if submitted:
            # Build a minimal row matching loaded data schema
            df_full = load_raw()
            row_base = df_full.drop(columns=["Disease"]).iloc[0:1].copy()
            # Override known fields
            if "Age" in row_base.columns:        row_base["Age"] = age
            if "Gender" in row_base.columns:     row_base["Gender"] = gender
            if "Smoker" in row_base.columns:     row_base["Smoker"] = smoker
            if "BloodPressure" in row_base.columns: row_base["BloodPressure"] = bp
            if "Cholesterol" in row_base.columns:   row_base["Cholesterol"] = chol
            if "InsuranceType" in row_base.columns: row_base["InsuranceType"] = instype

            pipe = fitted[model_choice]
            prob = pipe.predict_proba(row_base)[0][1] if hasattr(pipe, "predict_proba") else None
            pred = pipe.predict(row_base)[0]

            if prob is not None:
                col1, col2 = st.columns(2)
                col1.metric("Disease probability", f"{prob*100:.1f}%")
                col2.metric("Prediction", "🔴 Disease" if pred == 1 else "🟢 No disease")
                risk_color = "red" if prob > 0.6 else ("orange" if prob > 0.4 else "green")
                st.progress(int(prob * 100))
            else:
                st.metric("Prediction", "Disease" if pred == 1 else "No disease")
    else:
        st.info("Run training first (Tab 3).")