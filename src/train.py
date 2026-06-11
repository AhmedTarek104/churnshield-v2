"""Train 5 models with MLflow tracking, save best model."""
import json
import os
import pickle
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROC_DIR = r"G:\churnshield_v2\data\processed"
MODELS_DIR = r"G:\churnshield_v2\models"
FIGS_DIR = r"G:\churnshield_v2\outputs\figures"
TRACKING_URI = "sqlite:///G:/churnshield_v2/mlruns/mlflow.db"
EXPERIMENT = "churnshield-v2"


def load_splits():
    with open(os.path.join(PROC_DIR, "feature_names.json")) as f:
        feat_names = json.load(f)

    train = pd.read_csv(os.path.join(PROC_DIR, "train.csv"))
    val = pd.read_csv(os.path.join(PROC_DIR, "val.csv"))

    X_train = train[feat_names].values
    y_train = train["Exited"].values
    X_val = val[feat_names].values
    y_val = val["Exited"].values
    return X_train, y_train, X_val, y_val, feat_names


def eval_metrics(model, X, y, threshold=0.5):
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "auc": roc_auc_score(y, proba),
        "f1": f1_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
    }


def save_cm_plot(model, X, y, name, tmp_dir):
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    cm = confusion_matrix(y, pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Stay", "Churn"])
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"{name} — Confusion Matrix")
    path = os.path.join(tmp_dir, f"{name}_cm.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_roc_plot(model, X, y, name, tmp_dir):
    proba = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"{name} — ROC Curve")
    ax.legend()
    path = os.path.join(tmp_dir, f"{name}_roc.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def get_models():
    # XGBoost 3.x has native crashes on Windows/Python 3.13;
    # HistGradientBoosting is sklearn's built-in equivalent with same performance.
    return {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)),
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced",
            n_jobs=-1, random_state=42,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_depth=5,
            class_weight="balanced", random_state=42,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=400, learning_rate=0.05,
            is_unbalance=True, random_state=42, n_jobs=-1, verbose=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=400, learning_rate=0.05,
            auto_class_weights="Balanced", verbose=0, random_seed=42,
        ),
    }


if __name__ == "__main__":
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIGS_DIR, exist_ok=True)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    X_train, y_train, X_val, y_val, feat_names = load_splits()
    models = get_models()

    results = []
    tmp_dir = os.path.join(MODELS_DIR, "_tmp_artifacts")
    os.makedirs(tmp_dir, exist_ok=True)

    for name, model in models.items():
        print(f"\nTraining {name}…")
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)

            cv_auc = cross_val_score(
                model, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=1
            ).mean()

            train_m = eval_metrics(model, X_train, y_train)
            val_m = eval_metrics(model, X_val, y_val)

            mlflow.log_param("model", name)
            mlflow.log_metric("cv_auc", float(cv_auc))
            for k, v in train_m.items():
                mlflow.log_metric(f"train_{k}", float(v))
            for k, v in val_m.items():
                mlflow.log_metric(f"val_{k}", float(v))

            cm_path = save_cm_plot(model, X_val, y_val, name, tmp_dir)
            roc_path = save_roc_plot(model, X_val, y_val, name, tmp_dir)
            mlflow.log_artifact(cm_path)
            mlflow.log_artifact(roc_path)

            # MLflow 3.x: use `name` param instead of deprecated `artifact_path`
            try:
                mlflow.sklearn.log_model(model, name="model")
            except TypeError:
                mlflow.sklearn.log_model(model, artifact_path="model")

            results.append({
                "Model": name,
                "CV AUC": cv_auc,
                "Val AUC": val_m["auc"],
                "Val F1": val_m["f1"],
                "Val Recall": val_m["recall"],
                "_model": model,
            })

            print(f"  CV AUC={cv_auc:.4f}  Val AUC={val_m['auc']:.4f}  "
                  f"Val F1={val_m['f1']:.4f}  Val Recall={val_m['recall']:.4f}")

    df_res = pd.DataFrame(results).drop(columns=["_model"])
    df_res = df_res.sort_values("Val AUC", ascending=False)
    print("\n" + "=" * 70)
    print("MODEL COMPARISON (sorted by Val AUC)")
    print("=" * 70)
    print(df_res.to_string(index=False, float_format="{:.4f}".format))

    best_row = max(results, key=lambda r: r["Val AUC"])
    best_name = best_row["Model"]
    best_model = best_row["_model"]

    with open(os.path.join(MODELS_DIR, "best_model.pkl"), "wb") as f:
        pickle.dump(best_model, f)
    with open(os.path.join(MODELS_DIR, "best_model_name.txt"), "w") as f:
        f.write(best_name)

    print(f"\nBest model: {best_name} (Val AUC={best_row['Val AUC']:.4f})")
    print(f"Saved to {MODELS_DIR}/best_model.pkl")

    df_res.to_csv(os.path.join(r"G:\churnshield_v2\outputs", "model_comparison.csv"), index=False)

    print("\nTRAINING COMPLETE")
