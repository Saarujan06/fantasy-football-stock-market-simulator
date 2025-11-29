from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    classification_report,
    confusion_matrix,
)

import joblib


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEAN_DIR = PROJECT_ROOT / "data" / "cleaned_data"
RESULTS_DIR = PROJECT_ROOT / "results"

PRICES_FILE = CLEAN_DIR / "team_prices_current.csv"
TS_DATA_FILE = CLEAN_DIR / "timeseries_future_price_dataset.csv"


# -------------------------------------------------------------------
# 1. Load priced panel from Model 1
# -------------------------------------------------------------------
def load_priced_panel() -> pd.DataFrame:
    """
    Load the priced panel produced by model_current_price.py.

    Expected columns (at least):
        date, season, team, price, delta_P,
        pts, form3, xGD, opp_avg_pts, card_points, is_home, ...

    We keep everything and engineer time-series features on top.
    """
    if not PRICES_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find priced panel: {PRICES_FILE}\n"
            "Run model_current_price.py first."
        )

    df = pd.read_csv(PRICES_FILE, parse_dates=["date"])

    required = ["date", "season", "team", "price"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Priced panel is missing required columns: {missing}")

    # Sort for stable time ordering per team
    df = df.sort_values(["team", "date"]).reset_index(drop=True)
    return df


# -------------------------------------------------------------------
# 2. Build time-series dataset (X_t, y = price_{t+1}, direction)
# -------------------------------------------------------------------
def build_timeseries_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    From the priced panel, build a time-series supervised dataset where:

      - Each row corresponds to (team, match at time t)
      - Target for regression: price_next = price_{t+1}
      - Target for classification: direction (Up / Flat / Down)

    Features include:
      - price and lags
      - returns and rolling averages
      - match stats (pts, form3, xGD, opp_avg_pts, card_points, is_home)
    """
    print("Building time-series dataset for future price prediction...")

    all_rows = []

    # Work team by team to respect temporal structure
    for team, g in df.groupby("team"):
        g = g.sort_values("date").copy()

        # Price lags
        g["price_lag1"] = g["price"].shift(1)
        g["price_lag2"] = g["price"].shift(2)

        # Returns (simple diff)
        g["ret_t"] = g["price"] - g["price_lag1"]
        g["ret_t_lag1"] = g["ret_t"].shift(1)

        # Rolling stats over last 5 matches
        g["ma5_price"] = g["price"].rolling(window=5, min_periods=2).mean()
        g["ma5_ret"] = g["ret_t"].rolling(window=5, min_periods=2).mean()
        g["vol5_ret"] = g["ret_t"].rolling(window=5, min_periods=2).std()

        # Targets: next price & next delta
        g["price_next"] = g["price"].shift(-1)
        g["delta_next"] = g["price_next"] - g["price"]

        all_rows.append(g)

    ts = pd.concat(all_rows, ignore_index=True)

    # Threshold for "Flat" movement
    # Small absolute moves are considered flat.
    THRESH = 0.8
    ts["direction"] = np.where(
        ts["delta_next"] > THRESH,
        "Up",
        np.where(ts["delta_next"] < -THRESH, "Down", "Flat"),
    )

    # Drop rows where we don't know next price (last match of each team)
    ts = ts.dropna(subset=["price_next"]).reset_index(drop=True)

    # Save a copy for inspection
    TS_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    ts.to_csv(TS_DATA_FILE, index=False)
    print(f"✅ Saved time-series dataset → {TS_DATA_FILE}")
    print(f"Rows: {len(ts)}, teams: {ts['team'].nunique()}")

    return ts


# -------------------------------------------------------------------
# 3. Feature selection
# -------------------------------------------------------------------
def get_feature_columns(ts: pd.DataFrame) -> List[str]:
    """
    Choose which columns to use as features X_t.

    We include:
      - price & its lags
      - returns & rolling stats
      - match-level features that we know at time t
    """
    candidate_features = [
        # price dynamics
        "price",
        "price_lag1",
        "price_lag2",
        "ret_t",
        "ret_t_lag1",
        "ma5_price",
        "ma5_ret",
        "vol5_ret",
        # match-level / football stats
        "pts",
        "form3",
        "xGD",
        "opp_avg_pts",
        "card_points",
        "is_home",
    ]

    # Keep only those that actually exist
    feat_cols = [c for c in candidate_features if c in ts.columns]

    # Fill NaNs (early matches) with reasonable defaults
    ts[feat_cols] = ts[feat_cols].fillna(0.0)

    return feat_cols


# -------------------------------------------------------------------
# 4. Train / test split (time-based: last season as test)
# -------------------------------------------------------------------
def train_test_split_by_season(ts: pd.DataFrame) -> Tuple:
    """
    Use all seasons except the last as training data,
    and the final season as test.

    This respects time ordering and avoids look-ahead bias.
    """
    seasons = sorted(ts["season"].unique())
    test_season = seasons[-1]

    train_mask = ts["season"] < test_season
    test_mask = ts["season"] == test_season

    print(f"Train seasons: {sorted(ts.loc[train_mask, 'season'].unique())}")
    print(f"Test season:   {test_season}")

    return train_mask, test_mask


# -------------------------------------------------------------------
# 5. Train regression (price_next) + classification (direction)
# -------------------------------------------------------------------
def train_future_price_models() -> None:
    """
    Full pipeline:
      1. Load priced panel
      2. Build time-series dataset
      3. Select features
      4. Time-based train/test split
      5. Train regressor for price_next with TimeSeries CV
      6. Train classifier for direction (Up/Flat/Down)
      7. Save models + feature list
    """
    df = load_priced_panel()
    ts = build_timeseries_dataset(df)

    feat_cols = get_feature_columns(ts)
    print(f"Using features: {feat_cols}")

    X = ts[feat_cols].to_numpy(dtype=float)
    y_price = ts["price_next"].to_numpy(dtype=float)
    y_dir = ts["direction"].astype(str).to_numpy()

    train_mask, test_mask = train_test_split_by_season(ts)

    X_train, X_test = X[train_mask], X[test_mask]
    y_train_price, y_test_price = y_price[train_mask], y_price[test_mask]
    y_train_dir, y_test_dir = y_dir[train_mask], y_dir[test_mask]

    # ----------------------------
    # 5.1 Regression model
    # ----------------------------
    reg_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_depth=5,
                    learning_rate=0.08,
                    max_iter=350,
                    l2_regularization=0.01,
                    random_state=42,
                ),
            ),
        ]
    )

    print("\n=== Training regression model (price_next) ===")
    # TimeSeriesSplit on the training period
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(
        reg_pipeline,
        X_train,
        y_train_price,
        cv=tscv,
        scoring="r2",
        n_jobs=-1,
    )
    print(f"TimeSeries CV R² (train): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Fit on full training data
    reg_pipeline.fit(X_train, y_train_price)

    # Evaluate on train + test
    y_pred_train = reg_pipeline.predict(X_train)
    y_pred_test = reg_pipeline.predict(X_test)

    r2_train = r2_score(y_train_price, y_pred_train)
    mae_train = mean_absolute_error(y_train_price, y_pred_train)

    r2_test = r2_score(y_test_price, y_pred_test)
    mae_test = mean_absolute_error(y_test_price, y_pred_test)

    print("\nRegression performance:")
    print(f"  Train R²: {r2_train:.3f}, MAE: {mae_train:.3f}")
    print(f"  Test  R²: {r2_test:.3f}, MAE: {mae_test:.3f}")

    # ----------------------------
    # 5.2 Classification model (direction)
    # ----------------------------
    print("\n=== Training classification model (direction Up/Flat/Down) ===")

    # Reuse the scaler from the regression pipeline
    scaler: StandardScaler = reg_pipeline.named_steps["scaler"]
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=3,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    clf.fit(X_train_scaled, y_train_dir)

    y_dir_pred = clf.predict(X_test_scaled)

    print("\nDirection classification report (test season):")
    print(classification_report(y_test_dir, y_dir_pred, digits=3))

    print("Confusion matrix (rows = true, cols = predicted):")
    print(confusion_matrix(y_test_dir, y_dir_pred))

    # ----------------------------
    # 5.3 Save models and artifacts
    # ----------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(reg_pipeline, RESULTS_DIR / "future_price_regressor.joblib")
    joblib.dump(clf, RESULTS_DIR / "future_price_direction_classifier.joblib")
    joblib.dump(feat_cols, RESULTS_DIR / "future_price_features.joblib")

    print(f"\n✅ Saved regression + classification models to {RESULTS_DIR}")


# -------------------------------------------------------------------
# 6. Convenience: predict on the latest row for each team
# -------------------------------------------------------------------
def predict_next_week() -> pd.DataFrame:
    """
    Use the trained models to make a one-step-ahead prediction for each team,
    based on their most recent available match.

    Returns a DataFrame with:
        team, last_date, last_price, pred_price_next, pred_direction
    """
    if not (RESULTS_DIR / "future_price_regressor.joblib").exists():
        raise FileNotFoundError(
            "Models not found. Run this module once to train them:\n"
            "  python -m src.models.model_future_price"
        )

    df = load_priced_panel()
    ts = build_timeseries_dataset(df)
    feat_cols = joblib.load(RESULTS_DIR / "future_price_features.joblib")

    reg_pipeline = joblib.load(RESULTS_DIR / "future_price_regressor.joblib")
    clf = joblib.load(RESULTS_DIR / "future_price_direction_classifier.joblib")
    scaler: StandardScaler = reg_pipeline.named_steps["scaler"]

    latest_rows = (
        ts.sort_values("date")
        .groupby("team")
        .tail(1)
        .reset_index(drop=True)
    )

    X_latest = latest_rows[feat_cols].fillna(0.0).to_numpy(dtype=float)
    X_latest_scaled = scaler.transform(X_latest)

    pred_price_next = reg_pipeline.predict(X_latest)
    pred_dir = clf.predict(X_latest_scaled)

    out = pd.DataFrame(
        {
            "team": latest_rows["team"],
            "last_date": latest_rows["date"],
            "last_price": latest_rows["price"],
            "pred_price_next": pred_price_next,
            "pred_direction": pred_dir,
        }
    )

    out_file = RESULTS_DIR / "future_price_next_week_predictions.csv"
    out.to_csv(out_file, index=False)
    print(f"\n✅ Saved next-week predictions → {out_file}")

    return out


# -------------------------------------------------------------------
# CLI entry point
# -------------------------------------------------------------------
if __name__ == "__main__":
    # 1) Train models + print metrics
    train_future_price_models()

    # 2) Optional: also produce a next-week prediction file
    try:
        preds = predict_next_week()
        print("\nSample predictions:")
        print(preds.head())
    except Exception as e:
        print(f"\nCould not generate next-week predictions: {e}")