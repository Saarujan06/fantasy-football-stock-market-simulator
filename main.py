from __future__ import annotations

from pathlib import Path
import sys
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ============================================================================
# PATHS & PYTHON PATH FIX
# ============================================================================

# Project root is the folder where main.py lives
PROJECT_ROOT = Path(__file__).resolve().parent

# Ensure src/ is on sys.path so we can import our internal modules
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# ============================================================================
# IMPORTS FROM src/
# ============================================================================

from src.data_loader import load_and_split, FEATURE_COLS
from src.models import (
    train_logistic_regression,
    train_random_forest,
    train_knn,
    train_gradient_boosting,
)
from src.predict_future import predict_future_matches
from src.stock_direction import compute_stock_directions

# ============================================================================
# PLOTTING UTILITIES
# ============================================================================


def plot_feature_correlation(
    X: np.ndarray,
    feature_names: List[str],
    save_path: Path,
) -> None:
    """
    Plot a correlation heatmap between all features and save it as a PNG.
    """
    df = pd.DataFrame(X, columns=feature_names)
    corr = df.corr()

    plt.figure(figsize=(14, 10))
    im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im)
    plt.xticks(
        ticks=range(len(feature_names)),
        labels=feature_names,
        rotation=90,
    )
    plt.yticks(
        ticks=range(len(feature_names)),
        labels=feature_names,
    )
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_target_correlation(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    save_path: Path,
) -> None:
    """
    Plot correlation of each feature with the numeric target (0=loss,1=draw,2=win).
    Saves a bar plot as PNG.
    """
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    corr_with_y = df.corr()["target"].drop("target").sort_values()

    plt.figure(figsize=(10, 8))
    corr_with_y.plot(kind="barh")
    plt.xlabel("Correlation with match outcome (0=loss,1=draw,2=win)")
    plt.title("Feature–Target Correlation")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ============================================================================
# MODEL EVALUATION
# ============================================================================


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
) -> float:
    """
    Compute accuracy, classification report, confusion matrix,
    save all outputs inside results folder, and return accuracy.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # 1) Accuracy summary
    acc_df = pd.DataFrame({"model": [model_name], "accuracy": [acc]})
    acc_df.to_csv(RESULTS_DIR / f"{model_name}_accuracy.csv", index=False)

    # 2) Classification report (as CSV)
    report_dict = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(
        RESULTS_DIR / f"{model_name}_classification_report.csv",
        index=True,
    )

    # 3) Confusion matrix (heatmap)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix – {model_name}")
    plt.colorbar()
    plt.xticks([0, 1, 2], ["Loss (0)", "Draw (1)", "Win (2)"])
    plt.yticks([0, 1, 2], ["Loss (0)", "Draw (1)", "Win (2)"])

    # Annotate counts
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="black",
            )

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{model_name}_confusion_matrix.png")
    plt.close()

    print(f"{model_name} accuracy = {acc:.3f}")
    return acc


# ============================================================================
# MAIN WORKFLOW
# ============================================================================


def main() -> None:
    # ------------------------------------------------------------
    # 1) Load data & train/test split
    # ------------------------------------------------------------
    print("Loading train/test split...")
    X_train, X_test, y_train, y_test = load_and_split()

    # ------------------------------------------------------------
    # 2) Correlation diagnostics
    # ------------------------------------------------------------
    print("Plotting correlation diagnostics...")
    plot_feature_correlation(
        X_train,
        FEATURE_COLS,
        RESULTS_DIR / "feature_corr_heatmap.png",
    )
    plot_target_correlation(
        X_train,
        y_train,
        FEATURE_COLS,
        RESULTS_DIR / "feature_target_corr.png",
    )

    # ------------------------------------------------------------
    # 3) Train models
    # ------------------------------------------------------------
    # NOTE: individual training functions print their own "Training ..." messages.
    # Here we just print the best params again for clarity.

    # Logistic Regression (with StandardScaler & max_iter=1000 inside models.py)
    lr_model, lr_params = train_logistic_regression(X_train, y_train)
    print(f"Best LR params: {lr_params}")

    # Random Forest
    rf_model, rf_params = train_random_forest(X_train, y_train)
    print(f"Best RF params: {rf_params}")

    # KNN (with StandardScaler inside models.py)
    knn_model, knn_params = train_knn(X_train, y_train)
    print(f"Best KNN params: {knn_params}")

    # Gradient Boosting (NEW)
    gb_model, gb_params = train_gradient_boosting(X_train, y_train)
    print(f"Best GB params: {gb_params}")

    # ------------------------------------------------------------
    # 4) Evaluate models
    # ------------------------------------------------------------
    print("\nEvaluating models...")

    acc_lr = evaluate_model(lr_model, X_test, y_test, "LogReg")
    acc_rf = evaluate_model(rf_model, X_test, y_test, "RandomForest")
    acc_knn = evaluate_model(knn_model, X_test, y_test, "KNN")
    acc_gb = evaluate_model(gb_model, X_test, y_test, "GradBoost")

    scores = {
        "LogReg": acc_lr,
        "RandomForest": acc_rf,
        "KNN": acc_knn,
        "GradBoost": acc_gb,
    }

    # ------------------------------------------------------------
    # 5) Select best model by accuracy
    # ------------------------------------------------------------
    best_model_name = max(scores, key=scores.get)
    best_model = {
        "LogReg": lr_model,
        "RandomForest": rf_model,
        "KNN": knn_model,
        "GradBoost": gb_model,
    }[best_model_name]

    print("\n==============================")
    print(f"Best model: {best_model_name} (accuracy {scores[best_model_name]:.3f})")
    print("==============================")

    # ------------------------------------------------------------
    # 6) Predict FUTURE 2025–26 fixtures
    # ------------------------------------------------------------
    print("\nPredicting future 2025–26 fixtures...")
    future_pred_df = predict_future_matches(
        model=best_model,
        feature_cols=FEATURE_COLS,
        results_dir=RESULTS_DIR,
    )
    # predict_future_matches already prints and saves CSV, but we echo the path:
    print(
        "Saved future predictions to:",
        RESULTS_DIR / "future_predictions_2025_26.csv",
    )

    # ------------------------------------------------------------
    # 7) Compute STOCK DIRECTION from predictions
    # ------------------------------------------------------------
    print("\nComputing stock direction...")
    stock_summary = compute_stock_directions(
        future_predictions=future_pred_df,
        results_dir=RESULTS_DIR,
    )
    # stock_direction module saves its own CSV; we just echo:
    print(
        "Saved stock direction summary to:",
        RESULTS_DIR / "stock_direction_2025_26.csv",
    )

    print("\nAll tasks completed successfully.")


if __name__ == "__main__":
    main()