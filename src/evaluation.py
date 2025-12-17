from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def evaluate_classifier(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label: str,
    results_dir: str | Path,
) -> float:
    """
    Evaluate a trained classifier and save outputs into:
        <results_dir>/model_eval/

    Saves:
      - <label>_accuracy.csv
      - <label>_classification_report.csv
      - <label>_confusion_matrix.png

    Returns:
        Accuracy (float).
    """
    out_dir = _model_eval_dir(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))

    # --- classification report as CSV (best for marking) ---
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()
    report_path = out_dir / f"{label}_classification_report.csv"
    report_df.to_csv(report_path, index=True)

    # --- confusion matrix plot ---
    cm = confusion_matrix(y_test, y_pred)
    class_labels = _class_labels_from_y(y_test)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, interpolation="nearest")
    ax.set_title(f"Confusion Matrix — {label}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_yticks(np.arange(len(class_labels)))
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.tight_layout()
    cm_path = out_dir / f"{label}_confusion_matrix.png"
    fig.savefig(cm_path, bbox_inches="tight")
    plt.close(fig)

    # --- accuracy CSV ---
    acc_df = pd.DataFrame({"model": [label], "accuracy": [acc]})
    acc_path = out_dir / f"{label}_accuracy.csv"
    acc_df.to_csv(acc_path, index=False)

    print(f"{label} accuracy = {acc:.4f}")
    print(f"Saved: {report_path.name}, {cm_path.name}, {acc_path.name} (in {out_dir})")
    return acc


def save_accuracy_comparison(
    scores: Dict[str, float],
    results_dir: str | Path,
    save_plot: bool = True,
) -> Path:
    """
    Save overall model comparison into:
        <results_dir>/model_eval/accuracy_comparison.csv
        <results_dir>/model_eval/accuracy_comparison.png   (optional)
    """
    out_dir = _model_eval_dir(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = (
        pd.DataFrame({"model": list(scores.keys()), "accuracy": list(scores.values())})
        .sort_values("accuracy", ascending=False)
        .reset_index(drop=True)
    )
    csv_path = out_dir / "accuracy_comparison.csv"
    df.to_csv(csv_path, index=False)

    if save_plot:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(df["model"], df["accuracy"])
        ax.set_ylim(0, 1)
        ax.set_title("Accuracy Comparison")
        ax.set_ylabel("Accuracy")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        fig.tight_layout()
        fig.savefig(out_dir / "accuracy_comparison.png", bbox_inches="tight")
        plt.close(fig)

    print(f"Saved accuracy comparison to: {csv_path}")
    return csv_path


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _model_eval_dir(results_dir: str | Path) -> Path:
    return Path(results_dir) / "model_eval"


def _class_labels_from_y(y: np.ndarray) -> list[str]:
    # Your labels are 0/1/2 => show meaning in plots
    uniq = sorted(set(int(v) for v in np.unique(y)))
    mapping = {0: "0 (Loss)", 1: "1 (Draw)", 2: "2 (Win)"}
    return [mapping.get(v, str(v)) for v in uniq]