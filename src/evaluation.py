from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, ConfusionMatrixDisplay,
    RocCurveDisplay, PrecisionRecallDisplay, classification_report,
)

METRIC_FUNCTIONS = {
    "accuracy": lambda y_true, y_pred, y_proba: accuracy_score(y_true, y_pred),
    "precision": lambda y_true, y_pred, y_proba: precision_score(y_true, y_pred, zero_division=0),
    "recall": lambda y_true, y_pred, y_proba: recall_score(y_true, y_pred, zero_division=0),
    "f1": lambda y_true, y_pred, y_proba: f1_score(y_true, y_pred, zero_division=0),
    "roc_auc": lambda y_true, y_pred, y_proba: roc_auc_score(y_true, y_proba),
    "pr_auc": lambda y_true, y_pred, y_proba: average_precision_score(y_true, y_proba),
}

def evaluate_binary_classifier(model, X, y, threshold: float = 0.5) -> dict:
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    return {metric: func(y, y_pred, y_proba) for metric, func in METRIC_FUNCTIONS.items()}

def bootstrap_metric_distribution(y_true, y_proba, n_bootstrap: int = 1000,
                                  threshold: float = 0.5, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n = len(y_true)
    rows = []
    for i in range(n_bootstrap):
        idx = rng.choice(np.arange(n), size=n, replace=True)
        y_b = y_true[idx]
        p_b = y_proba[idx]
        pred_b = (p_b >= threshold).astype(int)
        row = {"bootstrap_iteration": i}
        for metric, func in METRIC_FUNCTIONS.items():
            try:
                row[metric] = func(y_b, pred_b, p_b)
            except ValueError:
                row[metric] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)

def summarize_bootstrap_metrics(bootstrap_df: pd.DataFrame, model_name: str,
                                metrics=None, ci: float = 95) -> pd.DataFrame:
    if metrics is None:
        metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    alpha = (100 - ci) / 2
    rows = []
    for metric in metrics:
        values = bootstrap_df[metric].dropna()
        rows.append({
            "model": model_name,
            "metric": metric,
            "mean": values.mean(),
            "std": values.std(ddof=1),
            "ci_lower": np.percentile(values, alpha),
            "ci_upper": np.percentile(values, 100 - alpha),
        })
    return pd.DataFrame(rows)

def format_ci_value(mean, lower, upper, decimals: int = 3) -> str:
    return f"{mean:.{decimals}f} [{lower:.{decimals}f}, {upper:.{decimals}f}]"

def make_formatted_metric_table(summary_df: pd.DataFrame, decimals: int = 3) -> pd.DataFrame:
    df = summary_df.copy()
    df["formatted"] = df.apply(
        lambda r: format_ci_value(r["mean"], r["ci_lower"], r["ci_upper"], decimals=decimals),
        axis=1,
    )
    table = df.pivot(index="model", columns="metric", values="formatted")
    order = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    return table[[c for c in order if c in table.columns]]

def classification_report_df(model, X, y, threshold: float = 0.5) -> pd.DataFrame:
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    return pd.DataFrame(classification_report(y, y_pred, output_dict=True, zero_division=0)).T

def plot_confusion_matrix(model, X, y, save_path=None, threshold: float = 0.5):
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y, y_pred)).plot(values_format="d")
    plt.title("Confusion Matrix")
    if save_path is not None:
        save_path = Path(save_path); save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_roc_curve(model, X, y, save_path=None):
    RocCurveDisplay.from_estimator(model, X, y)
    plt.title("ROC Curve")
    if save_path is not None:
        save_path = Path(save_path); save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_pr_curve(model, X, y, save_path=None):
    PrecisionRecallDisplay.from_estimator(model, X, y)
    plt.title("Precision-Recall Curve")
    if save_path is not None:
        save_path = Path(save_path); save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_metric_violins(bootstrap_long_df: pd.DataFrame, metric: str, save_path=None):
    plot_df = bootstrap_long_df[bootstrap_long_df["metric"] == metric].dropna(subset=["value"])
    models = list(plot_df["model"].unique())
    data = [plot_df.loc[plot_df["model"] == model, "value"].values for model in models]
    fig, ax = plt.subplots(figsize=(max(7, 1.8 * len(models)), 5))
    ax.violinplot(data, showmeans=True, showmedians=True)
    ax.set_xticks(np.arange(1, len(models) + 1))
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(f"Bootstrap distribution of {metric}")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path); save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
