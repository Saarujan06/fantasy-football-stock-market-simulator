from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEAN_DIR = PROJECT_ROOT / "data" / "cleaned_data"

FEATURES_FILE = CLEAN_DIR / "team_match_features.csv"
HIST_PRICES_FILE = CLEAN_DIR / "team_prices.csv"   # up to 2024–2025
OUT_FILE = CLEAN_DIR / "team_prices_2025.csv"

TARGET_SEASON = "2025-2026"

# Tuning parameters for smooth, stable price movements
K1 = 2.0     # weight on realised performance
K2 = 1.2     # weight on predicted performance
LAMBDA = 0.15  # scaling to keep volatility realistic


# ---------------------------------------------------------
# 1. Regression: predict pts from features
# ---------------------------------------------------------
def fit_regression(df: pd.DataFrame):
    """
    Fit linear regression:
        pts = β0 + β1*xGD + β2*card_points + β3*form3 + β4*opp_avg_pts + ε
    """
    features = ["xGD", "card_points", "form3", "opp_avg_pts"]

    train = df[df["season"] != TARGET_SEASON].dropna(subset=["pts"] + features)
    if train.empty:
        raise ValueError("No training data available to fit regression.")

    X = train[features].to_numpy(float)
    y = train["pts"].to_numpy(float)

    X_design = np.c_[np.ones(len(X)), X]       # add intercept
    coeffs, *_ = np.linalg.lstsq(X_design, y, rcond=None)

    print("\n=== Regression fitted on historical seasons ===")
    names = ["β0"] + features
    for n, c in zip(names, coeffs):
        print(f"{n:12s}: {c: .4f}")

    mean_pts = float(y.mean())
    print(f"\nHistorical mean points per match: {mean_pts:.4f}")

    return coeffs, mean_pts


def apply_regression(df: pd.DataFrame, coeffs):
    """Add pred_pts column = model prediction."""
    features = ["xGD", "card_points", "form3", "opp_avg_pts"]

    X = df[features].fillna(0.0).to_numpy(float)
    X_design = np.c_[np.ones(len(X)), X]

    df["pred_pts"] = X_design @ coeffs
    return df


# ---------------------------------------------------------
# 2. Starting prices: final observed price from 2024–2025
# ---------------------------------------------------------
def load_start_prices():
    if not HIST_PRICES_FILE.exists():
        raise FileNotFoundError(f"Missing historical price file: {HIST_PRICES_FILE}")

    df = pd.read_csv(HIST_PRICES_FILE, parse_dates=["date"])

    # isolate final price of 2024–2025 per team
    df_last = df.sort_values("date").groupby("team")["price"].last()

    league_mean = float(df_last.mean())

    print("\n=== Starting stock prices (end of 2024–2025) ===")
    print(df_last.sort_values(ascending=False).head())

    return df_last.to_dict(), league_mean


# ---------------------------------------------------------
# 3. Build 2025 pricing dataset
# ---------------------------------------------------------
def build_pricing_dataset_2025(save_to_csv=True):
    # --- Load full feature panel ---
    if not FEATURES_FILE.exists():
        raise FileNotFoundError(f"Missing features file: {FEATURES_FILE}")

    df = pd.read_csv(FEATURES_FILE, parse_dates=["date"])
    print(f"Loaded features: {len(df)} rows, {df.team.nunique()} teams")

    # --- Fit regression on older seasons ---
    coeffs, historic_mean = fit_regression(df)

    # --- Filter to 2025–2026 ---
    df25 = df[df["season"] == TARGET_SEASON].copy()
    df25 = df25.sort_values(["team", "date"]).reset_index(drop=True)

    if df25.empty:
        raise ValueError("No matches for season 2025–2026.")

    df25 = apply_regression(df25, coeffs)

    # --- Compute deviations from historical mean ---
    df25["dev_real"] = df25["pts"] - historic_mean
    df25["dev_pred"] = df25["pred_pts"] - historic_mean

    # Stable match-level delta
    df25["delta"] = (
        K1 * df25["dev_real"] +
        K2 * df25["dev_pred"]
    ) * LAMBDA

    # --- Build prices ---
    start_prices, mean_start = load_start_prices()

    def build_team_price(team_df):
        team = team_df["team"].iloc[0]
        P0 = start_prices.get(team, mean_start)

        team_df = team_df.sort_values("date").copy()
        team_df["price"] = P0 + team_df["delta"].cumsum()

        return team_df

    priced = df25.groupby("team", group_keys=False).apply(build_team_price)

    # Save
    if save_to_csv:
        priced.to_csv(OUT_FILE, index=False)
        print(f"\n✅ Saved 2025 pricing file → {OUT_FILE}")

    # Summary
    print("\n=== Final 2025 Price Summary ===")
    last = priced.sort_values("date").groupby("team")["price"].last()
    print("Range: £%.2f – £%.2f" % (last.min(), last.max()))
    print("\nTop:")
    print(last.sort_values(ascending=False).head())
    print("\nBottom:")
    print(last.sort_values().head())

    return priced


# ---------------------------------------------------------
# Script entry
# ---------------------------------------------------------
if __name__ == "__main__":
    df_price = build_pricing_dataset_2025(save_to_csv=True)
    print("\nSample:")
    print(df_price.head(20))