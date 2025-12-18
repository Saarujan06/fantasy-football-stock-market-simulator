from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def evaluate_classifier(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label: str,
    results_dir: str | Path,
    feature_names: Optional[list[str]] = None,
) -> float:
    """
    Evaluate a trained classifier and save outputs into:
        <results_dir>/model_eval/

    Saves:
      - <label>_accuracy.csv
      - <label>_classification_report.csv
      - <label>_confusion_matrix.png
      - <label>_confusion_matrix.csv
      - <label>_metrics_summary.csv
      - <label>_model_details.csv
      - <label>_coefficients.csv                (if available)
      - <label>_feature_importances.csv         (if available)
      - <label>_evaluation_report.pdf           (summary PDF)

    Returns:
        Accuracy (float).
    """
    out_dir = _model_eval_dir(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------
    # Predictions + metrics
    # -----------------------
    y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))

    # Macro/weighted metrics (useful for imbalanced classes)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    metrics_summary = pd.DataFrame(
        {
            "model": [label],
            "accuracy": [acc],
            "precision_macro": [float(prec_macro)],
            "recall_macro": [float(rec_macro)],
            "f1_macro": [float(f1_macro)],
            "precision_weighted": [float(prec_w)],
            "recall_weighted": [float(rec_w)],
            "f1_weighted": [float(f1_w)],
            "n_test": [int(len(y_test))],
            "n_features": [int(X_test.shape[1])],
        }
    )
    metrics_path = out_dir / f"{label}_metrics_summary.csv"
    metrics_summary.to_csv(metrics_path, index=False)

    # -----------------------
    # Classification report (CSV)
    # -----------------------
    report_dict = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_path = out_dir / f"{label}_classification_report.csv"
    report_df.to_csv(report_path, index=True)

    # -----------------------
    # Confusion matrix (CSV + PNG)
    # -----------------------
    uniq_classes = sorted(set(int(v) for v in np.unique(y_test)))
    class_labels = _class_labels_from_classes(uniq_classes)

    cm = confusion_matrix(y_test, y_pred, labels=uniq_classes)
    cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)

    cm_csv_path = out_dir / f"{label}_confusion_matrix.csv"
    cm_df.to_csv(cm_csv_path, index=True)

    cm_png_path = out_dir / f"{label}_confusion_matrix.png"
    _save_confusion_matrix_png(cm, class_labels, cm_png_path, title=f"Confusion Matrix — {label}")

    # -----------------------
    # Accuracy CSV (kept)
    # -----------------------
    acc_path = out_dir / f"{label}_accuracy.csv"
    pd.DataFrame({"model": [label], "accuracy": [acc]}).to_csv(acc_path, index=False)

    # -----------------------
    # Model details (params)
    # -----------------------
    details = {
        "label": label,
        "estimator_class": model.__class__.__name__,
    }
    try:
        params = model.get_params()
        for k, v in params.items():
            details[f"param__{k}"] = v
    except Exception:
        pass

    details_df = pd.DataFrame([details])
    details_path = out_dir / f"{label}_model_details.csv"
    details_df.to_csv(details_path, index=False)

    # -----------------------
    # Coefs / Importances (if available)
    # -----------------------
    coef_df = _extract_coefficients(model, feature_names, X_test.shape[1])
    coef_path = None
    if coef_df is not None and not coef_df.empty:
        coef_path = out_dir / f"{label}_coefficients.csv"
        coef_df.to_csv(coef_path, index=False)

    fi_df = _extract_feature_importances(model, feature_names, X_test.shape[1])
    fi_path = None
    if fi_df is not None and not fi_df.empty:
        fi_path = out_dir / f"{label}_feature_importances.csv"
        fi_df.to_csv(fi_path, index=False)

    # -----------------------
    # PDF report (one file)
    # -----------------------
    pdf_path = out_dir / f"{label}_evaluation_report.pdf"
    _write_pdf_report(
        pdf_path=pdf_path,
        label=label,
        metrics_df=metrics_summary,
        report_df=report_df,
        cm_df=cm_df,
        cm_array=cm,
        class_labels=class_labels,
        model_details_df=details_df,
        coef_df=coef_df,
        fi_df=fi_df,
    )

    print(f"{label} accuracy = {acc:.4f}")
    print(f"Saved outputs to: {out_dir}")
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
    # results_dir should be PROJECT_ROOT/results
    return Path(results_dir) / "model_eval"


def _class_labels_from_classes(classes: list[int]) -> list[str]:
    mapping = {0: "0 (Loss)", 1: "1 (Draw)", 2: "2 (Win)"}
    return [mapping.get(int(c), str(c)) for c in classes]


def _safe_feature_names(feature_names: Optional[list[str]], n_features: int) -> list[str]:
    if feature_names is not None and len(feature_names) == n_features:
        return feature_names
    return [f"x{i}" for i in range(n_features)]


def _extract_coefficients(
    model,
    feature_names: Optional[list[str]],
    n_features: int,
) -> Optional[pd.DataFrame]:
    """
    Linear models (e.g., LogisticRegression): coef_ can be (n_classes, n_features)
    """
    if not hasattr(model, "coef_"):
        return None

    coef = getattr(model, "coef_", None)
    if coef is None:
        return None
    coef = np.asarray(coef)

    feats = _safe_feature_names(feature_names, n_features)

    # Binary or OvR-like shapes:
    if coef.ndim == 1:
        df = pd.DataFrame({"feature": feats, "coef": coef})
        return df.reindex(df["coef"].abs().sort_values(ascending=False).index)

    # Multiclass: one column per class row
    out = {"feature": feats}
    for i in range(coef.shape[0]):
        out[f"coef_class_{i}"] = coef[i, :]

    df = pd.DataFrame(out)

    # Sort by strongest effect in any class
    coef_cols = [c for c in df.columns if c.startswith("coef_class_")]
    df["max_abs_coef"] = df[coef_cols].abs().max(axis=1)
    df = df.sort_values("max_abs_coef", ascending=False).drop(columns=["max_abs_coef"])
    return df


def _extract_feature_importances(
    model,
    feature_names: Optional[list[str]],
    n_features: int,
) -> Optional[pd.DataFrame]:
    """
    Tree-based models: feature_importances_
    """
    if not hasattr(model, "feature_importances_"):
        return None

    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return None

    imp = np.asarray(imp)
    feats = _safe_feature_names(feature_names, n_features)

    df = pd.DataFrame({"feature": feats, "importance": imp})
    return df.sort_values("importance", ascending=False)


def _save_confusion_matrix_png(
    cm: np.ndarray,
    class_labels: list[str],
    out_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
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
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _write_pdf_report(
    pdf_path: Path,
    label: str,
    metrics_df: pd.DataFrame,
    report_df: pd.DataFrame,
    cm_df: pd.DataFrame,
    cm_array: np.ndarray,
    class_labels: list[str],
    model_details_df: pd.DataFrame,
    coef_df: Optional[pd.DataFrame],
    fi_df: Optional[pd.DataFrame],
) -> None:
    """
    Create a readable multi-page PDF report using matplotlib tables + the CM plot.
    """
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    def add_table_page(title: str, df: pd.DataFrame, max_rows: int = 40) -> None:
        view = df.copy()
        if len(view) > max_rows:
            view = view.head(max_rows)

        fig, ax = plt.subplots(figsize=(11.69, 8.27))  # ~A4 landscape
        ax.axis("off")
        ax.set_title(title, fontsize=14, pad=12)

        tbl = ax.table(
            cellText=view.values,
            colLabels=view.columns,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.2)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    with PdfPages(pdf_path) as pdf:
        add_table_page(f"{label} — Metrics Summary", metrics_df)

        # classification report is big: reset index for readability
        rep_view = report_df.reset_index().rename(columns={"index": "metric"})
        add_table_page(f"{label} — Classification Report (Top)", rep_view, max_rows=40)

        # confusion matrix plot page
        fig, ax = plt.subplots(figsize=(8.27, 6.2))
        ax.imshow(cm_array, interpolation="nearest")
        ax.set_title(f"{label} — Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks(np.arange(len(class_labels)))
        ax.set_yticks(np.arange(len(class_labels)))
        ax.set_xticklabels(class_labels)
        ax.set_yticklabels(class_labels)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        for i in range(cm_array.shape[0]):
            for j in range(cm_array.shape[1]):
                ax.text(j, i, str(cm_array[i, j]), ha="center", va="center")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        cm_view = cm_df.reset_index().rename(columns={"index": "actual\\pred"})
        add_table_page(f"{label} — Confusion Matrix (Table)", cm_view, max_rows=20)

        # model details: transpose so it fits like a key/value table
        det_view = model_details_df.T.reset_index().rename(columns={"index": "field", 0: "value"})
        add_table_page(f"{label} — Model Details (Top)", det_view, max_rows=60)

        if coef_df is not None and not coef_df.empty:
            add_table_page(f"{label} — Coefficients (Top)", coef_df, max_rows=40)

        if fi_df is not None and not fi_df.empty:
            add_table_page(f"{label} — Feature Importances (Top)", fi_df, max_rows=40)