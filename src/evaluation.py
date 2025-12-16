from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


def evaluate_classifier(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label: str,
    save_dir: str | Path,
) -> float:
    """
    Evaluate a trained classifier on a test set and save:
      - classification report (.txt)
      - confusion matrix heatmap (.png)
      - accuracy summary (.csv)

    Args:
        model: fitted sklearn-style classifier with .predict(...)
        X_test: feature matrix (n_samples, n_features)
        y_test: true labels (n_samples,)
        label: short name for the model (used in filenames)
        save_dir: directory where results will be written

    Returns:
        Accuracy (float).
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # Predictions & accuracy
    # ---------------------------------------------------------------------
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Classification report (precision/recall/F1 per class)
    report = classification_report(y_test, y_pred)

    report_path = save_dir / f"{label}_classification_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Model: {label}\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report)

    # ---------------------------------------------------------------------
    # Confusion matrix plot
    # ---------------------------------------------------------------------
    cm = confusion_matrix(y_test, y_pred)
    classes = np.unique(y_test)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(f"Confusion matrix - {label}")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    # Tick labels as class names
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
            )

    fig.tight_layout()
    cm_path = save_dir / f"{label}_confusion_matrix.png"
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------------------
    # Accuracy summary table
    # ---------------------------------------------------------------------
    summary_df = pd.DataFrame(
        {
            "model": [label],
            "accuracy": [acc],
        }
    )
    summary_path = save_dir / f"{label}_accuracy.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"{label} accuracy = {acc:.4f}")
    print(f"Saved classification report to: {report_path}")
    print(f"Saved confusion matrix to: {cm_path}")
    print(f"Saved accuracy summary to: {summary_path}")

    return acc