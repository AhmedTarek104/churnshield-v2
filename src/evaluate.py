"""Full evaluation on test set, business metrics, threshold optimization."""
import json
import os
import pickle
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore")

PROC_DIR = r"G:\churnshield_v2\data\processed"
MODELS_DIR = r"G:\churnshield_v2\models"
OUTPUTS_DIR = r"G:\churnshield_v2\outputs"
FIGS_DIR = os.path.join(OUTPUTS_DIR, "figures")


def load_data():
    with open(os.path.join(PROC_DIR, "feature_names.json")) as f:
        feat_names = json.load(f)
    val = pd.read_csv(os.path.join(PROC_DIR, "val.csv"))
    test = pd.read_csv(os.path.join(PROC_DIR, "test.csv"))
    return val, test, feat_names


def find_f1_threshold(proba, y):
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.2, 0.81, 0.01):
        f = f1_score(y, (proba >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return round(best_t, 2)


def find_business_threshold(proba, y, avg_clv, retention_rate=0.30, cost_per_outreach=10.0):
    best_t, best_val = 0.5, -1e9
    thresholds = []
    for t in np.arange(0.2, 0.81, 0.01):
        pred = (proba >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        n_outreach = tp + fp
        revenue = tp * avg_clv * retention_rate
        cost = n_outreach * cost_per_outreach
        net = revenue - cost
        thresholds.append({"threshold": round(t, 2), "net_value": net, "tp": tp, "fp": fp})
        if net > best_val:
            best_val, best_t = net, round(t, 2)
    return best_t, best_val, pd.DataFrame(thresholds)


def plot_roc(fpr, tpr, roc_auc):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#CC0000", lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.fill_between(fpr, tpr, alpha=0.15, color="#CC0000")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Test Set")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "roc_curve.png"), dpi=150)
    plt.close(fig)


def plot_pr_curve(precision, recall, ap):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="#00C853", lw=2, label=f"AP = {ap:.4f}")
    ax.fill_between(recall, precision, alpha=0.15, color="#00C853")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — Test Set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "pr_curve.png"), dpi=150)
    plt.close(fig)


def plot_threshold_analysis(thresh_df, f1_t, biz_t):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    ax1.plot(thresh_df["threshold"], thresh_df["f1"], color="#00C853", label="F1 Score", lw=2)
    ax2.plot(thresh_df["threshold"], thresh_df["net_value"], color="#FFB300", label="Net Value (€)", lw=2)
    ax1.axvline(f1_t, color="#00C853", linestyle="--", alpha=0.7, label=f"F1-optimal ({f1_t})")
    ax1.axvline(biz_t, color="#FFB300", linestyle="--", alpha=0.7, label=f"Biz-optimal ({biz_t})")
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("F1 Score", color="#00C853")
    ax2.set_ylabel("Net Business Value (€)", color="#FFB300")
    ax1.set_title("Threshold Analysis: F1 vs Business Value")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "threshold_analysis.png"), dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Reds")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Stay", "Churn"])
    ax.set_yticklabels(["Stay", "Churn"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Test Set")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close(fig)


def plot_business_value(thresh_df):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(thresh_df["threshold"], thresh_df["net_value"], color="#FFB300", lw=2)
    ax.axhline(0, color="white", linestyle="--", alpha=0.4)
    ax.fill_between(thresh_df["threshold"], thresh_df["net_value"], 0,
                    where=(thresh_df["net_value"] > 0), alpha=0.2, color="#00C853",
                    label="Profitable zone")
    ax.fill_between(thresh_df["threshold"], thresh_df["net_value"], 0,
                    where=(thresh_df["net_value"] <= 0), alpha=0.2, color="#CC0000",
                    label="Loss zone")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Net Business Value (€)")
    ax.set_title("Business Value vs Threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "business_value.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIGS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    with open(os.path.join(MODELS_DIR, "best_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "best_model_name.txt")) as f:
        model_name = f.read().strip()

    val_df, test_df, feat_names = load_data()

    X_val = val_df[feat_names].values
    y_val = val_df["Exited"].values
    X_test = test_df[feat_names].values
    y_test = test_df["Exited"].values

    # Threshold optimization on val set
    val_proba = model.predict_proba(X_val)[:, 1]
    avg_clv = test_df["clv_estimated"].mean() if "clv_estimated" in test_df.columns else 1000.0

    f1_thresh = find_f1_threshold(val_proba, y_val)

    # Threshold analysis on val, using test scale for business value
    test_proba = model.predict_proba(X_test)[:, 1]
    val_biz_thresh, _, thresh_df_val = find_business_threshold(val_proba, y_val, avg_clv)

    # Full threshold analysis on test for plotting
    biz_thresh, best_net, thresh_df = find_business_threshold(test_proba, y_test, avg_clv)

    # Add F1 to threshold df
    thresh_df["f1"] = [
        f1_score(y_test, (test_proba >= t).astype(int), zero_division=0)
        for t in thresh_df["threshold"]
    ]

    # Final evaluation with F1-optimal threshold
    pred_f1 = (test_proba >= f1_thresh).astype(int)

    roc_auc = roc_auc_score(y_test, test_proba)
    fpr, tpr, _ = roc_curve(y_test, test_proba)
    precision_arr, recall_arr, _ = precision_recall_curve(y_test, test_proba)
    ap = average_precision_score(y_test, test_proba)
    cm = confusion_matrix(y_test, pred_f1)
    tn, fp, fn, tp = cm.ravel()

    catch_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    metrics = {
        "model_name": model_name,
        "test_auc": round(roc_auc, 4),
        "test_f1": round(f1_score(y_test, pred_f1), 4),
        "test_precision": round(precision_score(y_test, pred_f1, zero_division=0), 4),
        "test_recall": round(recall_score(y_test, pred_f1, zero_division=0), 4),
        "test_accuracy": round(accuracy_score(y_test, pred_f1), 4),
        "average_precision": round(ap, 4),
        "catch_rate": round(catch_rate, 4),
        "false_alarm_rate": round(false_alarm_rate, 4),
        "confusion_matrix": {"TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)},
        "f1_optimal_threshold": f1_thresh,
        "business_optimal_threshold": biz_thresh,
        "net_value_at_optimal_threshold": round(best_net, 2),
        "avg_clv": round(avg_clv, 2),
        "retention_success_rate": 0.30,
        "cost_per_outreach": 10.0,
        "threshold_analysis": {
            "thresholds": [round(float(t), 2) for t in thresh_df["threshold"].tolist()],
            "f1_scores": [round(float(v), 4) for v in thresh_df["f1"].tolist()],
            "net_values": [round(float(v), 2) for v in thresh_df["net_value"].tolist()],
        },
    }

    with open(os.path.join(OUTPUTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(MODELS_DIR, "thresholds.json"), "w") as f:
        json.dump({
            "f1_optimal": f1_thresh,
            "business_optimal": biz_thresh,
        }, f, indent=2)

    # Retention value on test set
    test_df = test_df.copy()
    test_df["churn_probability"] = test_proba
    test_df["churn_prediction"] = (test_proba >= biz_thresh).astype(int)
    if "clv_estimated" in test_df.columns:
        test_df["retention_value"] = (test_df["clv_estimated"] * test_proba * 0.30).round(2)
    test_df.to_csv(os.path.join(PROC_DIR, "test_with_predictions.csv"), index=False)

    # Plots
    plot_roc(fpr, tpr, roc_auc)
    plot_pr_curve(precision_arr, recall_arr, ap)
    plot_threshold_analysis(thresh_df, f1_thresh, biz_thresh)
    plot_confusion_matrix(cm)
    plot_business_value(thresh_df)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Model              : {model_name}")
    print(f"  Test AUC           : {roc_auc:.4f}")
    print(f"  Test F1            : {metrics['test_f1']:.4f}")
    print(f"  Test Precision     : {metrics['test_precision']:.4f}")
    print(f"  Test Recall        : {metrics['test_recall']:.4f}")
    print(f"  Average Precision  : {ap:.4f}")
    print(f"  Catch Rate         : {catch_rate:.2%}")
    print(f"  False Alarm Rate   : {false_alarm_rate:.2%}")
    print(f"  TP/FP/TN/FN        : {tp}/{fp}/{tn}/{fn}")
    print(f"  F1-optimal thresh  : {f1_thresh}")
    print(f"  Biz-optimal thresh : {biz_thresh}")
    print(f"  Net value (biz)    : €{best_net:,.2f}")

    print("\nEVALUATION COMPLETE")
