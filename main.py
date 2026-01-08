from __future__ import annotations

from pathlib import Path
import sys

# ============================================================================
# PATH SETUP
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# IMPORTS
# ============================================================================
from src.pipeline.build_features import build as build_features
from src.price_engine.pricing_engine import run_pricing_engine

from src.data_loader import load_and_split, FEATURE_COLS
from src.models import (
    train_logistic_regression,
    train_random_forest,
    train_knn,
    train_gradient_boosting,
)

from src.predict_future import predict_future_matches
from src.stock_direction import compute_stock_directions
from src.evaluation import evaluate_classifier

try:
    from src.graphs.plot_stock_chart import plot_stock_chart
except Exception as e:
    plot_stock_chart = None
    _PLOT_IMPORT_ERROR = e

try:
    from src.graphs.plot_stock_signals import plot_stock_signals
except Exception as e:
    plot_stock_signals = None
    _SIGNALS_IMPORT_ERROR = e


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main() -> None:
    # ------------------------------------------------------------
    # 0) Rebuild features (season is ongoing, so keep this up to date)
    # ------------------------------------------------------------
    print("\n[0/6] Building cleaned features (team_match_features.csv)...")
    build_features()

    # ------------------------------------------------------------
    # 1) Pricing engine (it writes into results/pricing_engine/)
    # ------------------------------------------------------------
    print("\n[1/6] Running pricing engine...")
    run_pricing_engine()

    # ------------------------------------------------------------
    # 2) Load data & split for ML
    # ------------------------------------------------------------
    print("\n[2/6] Loading historical data...")
    X_train, X_test, y_train, y_test = load_and_split()

    # ------------------------------------------------------------
    # 3) Train models
    # ------------------------------------------------------------
    print("\n[3/6] Training models...")
    lr_model, _ = train_logistic_regression(X_train, y_train)
    rf_model, _ = train_random_forest(X_train, y_train)
    knn_model, _ = train_knn(X_train, y_train)
    gb_model, _ = train_gradient_boosting(X_train, y_train)

    # ------------------------------------------------------------
    # 4) Evaluate models + select best
    # ------------------------------------------------------------
    print("\n[4/6] Evaluating models...")
    scores = {
        "LogReg": evaluate_classifier(
            lr_model, X_test, y_test, "LogReg", RESULTS_DIR, feature_names=FEATURE_COLS
        ),
        "RandomForest": evaluate_classifier(
            rf_model, X_test, y_test, "RandomForest", RESULTS_DIR, feature_names=FEATURE_COLS
        ),
        "KNN": evaluate_classifier(
            knn_model, X_test, y_test, "KNN", RESULTS_DIR, feature_names=FEATURE_COLS
        ),
        "GradBoost": evaluate_classifier(
            gb_model, X_test, y_test, "GradBoost", RESULTS_DIR, feature_names=FEATURE_COLS
        ),
    }

    best_model_name = max(scores, key=scores.get)
    best_model = {
        "LogReg": lr_model,
        "RandomForest": rf_model,
        "KNN": knn_model,
        "GradBoost": gb_model,
    }[best_model_name]

    print("\n==============================")
    print(f"Best model: {best_model_name} ({scores[best_model_name]:.4f})")
    print("==============================")

    # ------------------------------------------------------------
    # 5) Predict next matchweek
    # ------------------------------------------------------------
    print("\n[5/6] Predicting next matchweek (2025–26)...")
    future_pred_df = predict_future_matches(
        model=best_model,
        feature_cols=FEATURE_COLS,
    )

    # ------------------------------------------------------------
    # 6) Stock signals + charts
    # ------------------------------------------------------------
    print("\n[6/6] Computing stock direction...")
    stock_df = compute_stock_directions(
        future_predictions=future_pred_df,
    )

    print("\nGenerating signal ranking chart...")
    if plot_stock_signals is None:
        print("Skipping signal chart because plot_stock_signals could not be imported.")
        print(f"Import error was: {_SIGNALS_IMPORT_ERROR}")
    else:
        plot_stock_signals()

    print("\nGenerating interactive stock chart...")
    if plot_stock_chart is None:
        print("Skipping chart generation because plot_stock_chart could not be imported.")
        print(f"Import error was: {_PLOT_IMPORT_ERROR}")
        print("Fix by installing plotly:  pip install plotly kaleido")
    else:
        plot_stock_chart(auto_open=True)

    print("\n✅ Pipeline completed successfully.")


if __name__ == "__main__":
    main()