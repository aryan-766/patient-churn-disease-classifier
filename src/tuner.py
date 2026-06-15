"""
tuner.py
GridSearchCV, RandomizedSearchCV, and Optuna for classification.
"""
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def grid_search_xgb(preprocessor, X_train, y_train):
    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", XGBClassifier(random_state=42, verbosity=0, eval_metric="logloss")),
    ])
    param_grid = {
        "model__n_estimators":   [100, 300],
        "model__max_depth":      [3, 5, 7],
        "model__learning_rate":  [0.05, 0.1, 0.2],
        "model__subsample":      [0.7, 0.9],
    }
    gs = GridSearchCV(pipe, param_grid, cv=CV, scoring="roc_auc", n_jobs=-1, verbose=1)
    gs.fit(X_train, y_train)
    print(f"[grid_search_xgb] Best AUC: {gs.best_score_:.4f} | {gs.best_params_}")
    return gs


def random_search_rf(preprocessor, X_train, y_train, n_iter=20):
    from scipy.stats import randint
    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)),
    ])
    param_dist = {
        "model__n_estimators": randint(100, 500),
        "model__max_depth":    randint(3, 20),
        "model__min_samples_split": randint(2, 15),
        "model__min_samples_leaf":  randint(1, 8),
    }
    rs = RandomizedSearchCV(pipe, param_dist, n_iter=n_iter, cv=CV,
                             scoring="roc_auc", n_jobs=-1, random_state=42)
    rs.fit(X_train, y_train)
    print(f"[random_search_rf] Best AUC: {rs.best_score_:.4f} | {rs.best_params_}")
    return rs


def optuna_xgb(preprocessor, X_train, y_train, n_trials=40):
    from sklearn.model_selection import cross_val_score

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
            "max_depth":        trial.suggest_int("max_depth", 3, 9),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.5, 5),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }
        pipe = Pipeline([
            ("prep", preprocessor),
            ("model", XGBClassifier(**params, random_state=42, verbosity=0, eval_metric="logloss")),
        ])
        return cross_val_score(pipe, X_train, y_train, cv=CV, scoring="roc_auc", n_jobs=-1).mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"[optuna_xgb] Best AUC: {study.best_value:.4f}")
    best_pipe = Pipeline([
        ("prep", preprocessor),
        ("model", XGBClassifier(**study.best_params, random_state=42, verbosity=0, eval_metric="logloss")),
    ])
    best_pipe.fit(X_train, y_train)
    return best_pipe, study