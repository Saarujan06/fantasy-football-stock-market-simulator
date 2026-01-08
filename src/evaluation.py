from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


# =============================================================================
# Public API
# =============================================================================

def evaluate_classifier(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label: str,
    results_dir: str | Path,
    feature_names: Optional[list[str]] = None,
    max_features_table: int = 25,
) -> float:

    results_dir = Path(results_dir)
    out_dir = results_dir / "model_eval" / label
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{label}_evaluation.pdf"

    # ---- predictions + metrics ----
    y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))

    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    # ---- report tables ----
    report_df = (
        pd.DataFrame(
            classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        )
        .transpose()
        .reset_index()
        .rename(columns={"index": "class_or_avg"})
    )

    uniq = sorted(set(int(v) for v in np.unique(y_test)))
    class_labels = _class_labels_from_y(y_test)
    cm = confusion_matrix(y_test, y_pred, labels=uniq)

    dom_err = _dominant_misclassification(cm, class_labels)

    summary_df = pd.DataFrame(
        {
            "Metric": [
                "Test observations",
                "Accuracy",
                "Balanced accuracy",
                "Macro precision",
                "Macro recall",
                "Macro F1",
                "Weighted F1",
                "Dominant error",
            ],
            "Value": [
                int(len(y_test)),
                f"{acc:.4f}",
                f"{bal_acc:.4f}",
                f"{float(prec_macro):.4f}",
                f"{float(rec_macro):.4f}",
                f"{float(f1_macro):.4f}",
                f"{float(f1_w):.4f}",
                dom_err,
            ],
        }
    )

    metrics_df = pd.DataFrame(
        {
            "metric": [
                "accuracy",
                "balanced_accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
                "precision_weighted",
                "recall_weighted",
                "f1_weighted",
                "n_test",
            ],
            "value": [
                acc,
                bal_acc,
                float(prec_macro),
                float(rec_macro),
                float(f1_macro),
                float(prec_w),
                float(rec_w),
                float(f1_w),
                int(len(y_test)),
            ],
        }
    )

    # ---- overview + optional explainability ----
    est = _unwrap_estimator(model)
    overview_df = _model_overview_table(est, label)

    safe_feats = _safe_feature_names(feature_names, _infer_n_features(est, X_test))
    coef_df = _extract_coefficients(est, safe_feats)
    fi_df = _extract_feature_importances(est, safe_feats)

    # ---- write PDF only ----
    with PdfPages(pdf_path) as pdf:
        _pdf_title_page(pdf, label=label, overview_df=overview_df)
        _pdf_table_page(
            pdf,
            title=f"{label} — SUMMARY OUTPUT (Classification)",
            df=summary_df,
            fontsize=12,
            cell_loc="left",
        )
        _pdf_table_page(pdf, title=f"{label} — Metrics Summary", df=metrics_df, fontsize=10)
        _pdf_table_page(
            pdf,
            title=f"{label} — Classification Report",
            df=report_df,
            fontsize=8,
            max_rows=50,
        )
        _pdf_confusion_matrix_page(
            pdf,
            title=f"{label} — Confusion Matrix",
            cm=cm,
            class_labels=class_labels,
        )

        if coef_df is not None and not coef_df.empty:
            _pdf_table_page(
                pdf,
                title=f"{label} — Coefficients (Top {min(max_features_table, len(coef_df))})",
                df=coef_df.head(max_features_table),
                fontsize=8,
                max_rows=max_features_table + 1,
            )

        if fi_df is not None and not fi_df.empty:
            _pdf_table_page(
                pdf,
                title=f"{label} — Feature Importances (Top {min(max_features_table, len(fi_df))})",
                df=fi_df.head(max_features_table),
                fontsize=8,
                max_rows=max_features_table + 1,
            )

    print(f"{label} accuracy = {acc:.4f}")
    print(f"Saved model evaluation PDF: {pdf_path}")
    return acc


def save_accuracy_comparison(scores: Dict[str, float], results_dir: str | Path) -> Path:

    results_dir = Path(results_dir)
    out_dir = results_dir / "model_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = (
        pd.DataFrame({"model": list(scores.keys()), "accuracy": list(scores.values())})
        .sort_values("accuracy", ascending=False)
        .reset_index(drop=True)
    )

    pdf_path = out_dir / "accuracy_comparison.pdf"
    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.bar(df["model"], df["accuracy"])
        ax.set_ylim(0, 1)
        ax.set_title("Accuracy Comparison")
        ax.set_ylabel("Accuracy")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        _pdf_table_page(pdf, title="Accuracy Comparison (Table)", df=df, fontsize=10, max_rows=50)

    print(f"Saved accuracy comparison PDF: {pdf_path}")
    return pdf_path


# =============================================================================
# Helpers
# =============================================================================

def _class_labels_from_y(y: np.ndarray) -> list[str]:
    uniq = sorted(set(int(v) for v in np.unique(y)))
    mapping = {0: "0 (Loss)", 1: "1 (Draw)", 2: "2 (Win)"}
    return [mapping.get(v, str(v)) for v in uniq]


def _dominant_misclassification(cm: np.ndarray, class_labels: list[str]) -> str:
    cm_off = cm.copy().astype(int)
    np.fill_diagonal(cm_off, 0)

    max_err = int(cm_off.max()) if cm_off.size else 0
    if max_err <= 0:
        return "None (perfect)"

    i, j = np.unravel_index(int(cm_off.argmax()), cm_off.shape)
    return f"{class_labels[i]} → {class_labels[j]} [{max_err}]"


def _safe_feature_names(feature_names: Optional[list[str]], n_features: int) -> list[str]:
    if feature_names is not None and len(feature_names) == n_features:
        return feature_names
    return [f"x{i}" for i in range(n_features)]


def _unwrap_estimator(model):
    if hasattr(model, "named_steps") and isinstance(getattr(model, "named_steps"), dict):
        try:
            return list(model.named_steps.values())[-1]
        except Exception:
            return model
    return model


def _infer_n_features(estimator, X_test: np.ndarray) -> int:
    if hasattr(estimator, "coef_"):
        coef = np.asarray(getattr(estimator, "coef_", None))
        if coef is not None and coef.size > 0:
            return int(coef.shape[-1])
    if hasattr(estimator, "feature_importances_"):
        imp = np.asarray(getattr(estimator, "feature_importances_", None))
        if imp is not None and imp.size > 0:
            return int(len(imp))
    return int(X_test.shape[1])


def _model_overview_table(estimator, label: str) -> pd.DataFrame:
    rows = [
        {"field": "model_label", "value": label},
        {"field": "estimator_class", "value": estimator.__class__.__name__},
    ]

    try:
        params = estimator.get_params()
    except Exception:
        params = {}

    def is_simple(v: Any) -> bool:
        return isinstance(v, (str, int, float, bool, type(None)))

    picked = []
    for k, v in params.items():
        if "__" in k:
            continue
        if is_simple(v):
            picked.append((k, v))

    for k, v in picked[:25]:
        rows.append({"field": f"param.{k}", "value": str(v)})

    return pd.DataFrame(rows)


def _extract_coefficients(estimator, feature_names: list[str]) -> Optional[pd.DataFrame]:
    if not hasattr(estimator, "coef_"):
        return None
    coef = getattr(estimator, "coef_", None)
    if coef is None:
        return None

    coef = np.asarray(coef)
    if coef.ndim == 1:
        df = pd.DataFrame({"feature": feature_names, "coef": coef})
        df["abs_coef"] = np.abs(df["coef"])
        return (
            df.sort_values("abs_coef", ascending=False)
            .drop(columns=["abs_coef"])
            .reset_index(drop=True)
        )

    out = {"feature": feature_names}
    for i in range(coef.shape[0]):
        out[f"coef_class_{i}"] = coef[i, :]

    df = pd.DataFrame(out)
    coef_cols = [c for c in df.columns if c.startswith("coef_class_")]
    df["max_abs_coef"] = df[coef_cols].abs().max(axis=1)
    return (
        df.sort_values("max_abs_coef", ascending=False)
        .drop(columns=["max_abs_coef"])
        .reset_index(drop=True)
    )


def _extract_feature_importances(estimator, feature_names: list[str]) -> Optional[pd.DataFrame]:
    if not hasattr(estimator, "feature_importances_"):
        return None
    imp = getattr(estimator, "feature_importances_", None)
    if imp is None:
        return None

    imp = np.asarray(imp)
    df = pd.DataFrame({"feature": feature_names, "importance": imp})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def _pdf_title_page(pdf: PdfPages, label: str, overview_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")
    ax.set_title(f"{label} — Model Evaluation", fontsize=18, pad=20)

    tbl = ax.table(
        cellText=overview_df.values,
        colLabels=overview_df.columns,
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.4)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _pdf_table_page(
    pdf: PdfPages,
    title: str,
    df: pd.DataFrame,
    fontsize: int = 9,
    max_rows: int = 35,
    cell_loc: str = "center",
) -> None:
    view = df if len(df) <= max_rows else df.head(max_rows)

    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")
    ax.set_title(title, fontsize=14, pad=14)

    tbl = ax.table(
        cellText=view.values,
        colLabels=view.columns,
        loc="center",
        cellLoc=cell_loc,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.scale(1.0, 1.25)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _pdf_confusion_matrix_page(
    pdf: PdfPages,
    title: str,
    cm: np.ndarray,
    class_labels: list[str],
) -> None:
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.set_title(title, fontsize=14, pad=14)

    # Lighter, report-friendly colours (no PNG output)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", alpha=0.85)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_yticks(np.arange(len(class_labels)))
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    # Always-readable annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)