from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


# =============================================================================
# PATHS & GLOBAL CONFIG
# =============================================================================

# This file lives in src/price_engine/, so parents[2] is the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "cleaned_data"
MATCH_FILE = DATA_DIR / "team_match_features.csv"

RESULTS_DIR = PROJECT_ROOT / "results" / "pricing_engine"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TIMESERIES_OUT = RESULTS_DIR / "price_timeseries.csv"
CURRENT_OUT = RESULTS_DIR / "current_prices_2025_2026.csv"

# Seasons and history window
HISTORY_START_YEAR = 2010          # include seasons 2010_2011 and onwards
FINAL_SEASON = "2025_2026"         # season we treat as "current"

# Price dynamics
BASE_PRICE = 100.0                 # starting price when a team enters the window
TARGET_DELTA_STD = 2.0             # target std.dev. for ΔPrice_t
RECENCY_LAMBDA = 0.92              # season recency decay factor (0 < λ < 1)

# Final price scaling range (you can tweak these)
TARGET_MIN_PRICE = 50.0
TARGET_MAX_PRICE = 1000.0

# MANUAL weights for match-level features (our design choice)
W_PTS = 1.0       # match result points (3, 1, 0)
W_FORM = 0.8      # short-term form (rolling 3-game average)
W_XGD = 0.7       # expected goal difference (or goal_diff fallback)
W_CARDS = 0.3     # card_points (penalises ill-discipline)
W_OPP = 0.5       # opponent average points (tougher opposition)


# =============================================================================
# 1. LOAD MATCH FEATURES
# =============================================================================

def load_match_features() -> pd.DataFrame:
    """
    Load engineered match-level features and filter to seasons >= HISTORY_START_YEAR.

    Expected columns (from ETL build_features.py):
        - season          (e.g. '2010_2011')
        - date
        - team
        - result          ('W','D','L')
        - pts             (3,1,0)
        - form3
        - xGD             (or goal_diff as fallback)
        - card_points
        - opp_avg_pts
    """
    if not MATCH_FILE.exists():
        raise FileNotFoundError(
            f"{MATCH_FILE} not found.\n"
            "Run the ETL pipeline first, e.g.:\n"
            "    python -m src.pipeline.build_features"
        )

    df = pd.read_csv(MATCH_FILE)
    df.columns = [c.strip() for c in df.columns]

    # Extract season start year: "2010_2011" → 2010
    if "season" not in df.columns:
        raise ValueError("Expected a 'season' column in team_match_features.csv")

    df["season_year"] = (
        df["season"].astype(str).str.split("_").str[0].astype("Int64")
    )

    df = df[df["season_year"] >= HISTORY_START_YEAR].copy()

    # Ensure date is datetime
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Keep valid results only
    if "result" in df.columns:
        df = df[df["result"].isin(["W", "D", "L"])].copy()

    # Ensure we have the numeric features we need
    numeric_cols = ["pts", "form3", "xGD", "card_points", "opp_avg_pts"]

    # If xGD missing, fall back to goal_diff if available
    if "xGD" not in df.columns and "goal_diff" in df.columns:
        df["xGD"] = df["goal_diff"]

    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0.0

    df[numeric_cols] = (
        df[numeric_cols].apply(lambda s: pd.to_numeric(s, errors="coerce")).fillna(0.0)
    )

    df = df.sort_values(["team", "season_year", "date"]).reset_index(drop=True)
    return df


# =============================================================================
# 2. MANUAL ΔPRICE FORMULA
# =============================================================================

def compute_delta_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-match price change ΔPrice_t using MANUAL, football-based weights.

    We first standardise each factor so weights are comparable, then apply:

        ΔP_raw = W_PTS  * pts_z
               + W_FORM * form3_z
               + W_XGD  * xGD_z
               - W_CARDS * card_points_z
               - W_OPP   * opp_avg_pts_z

    Finally we rescale ΔP_raw so that std(ΔP) ≈ TARGET_DELTA_STD.

    This is the *designed* mapping from performance to price — not ML.
    """
    df = df.copy()

    feature_cols = ["pts", "form3", "xGD", "card_points", "opp_avg_pts"]

    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols])
    pts_z, form3_z, xGD_z, cards_z, opp_z = X.T

    delta_raw = (
        W_PTS * pts_z
        + W_FORM * form3_z
        + W_XGD * xGD_z
        - W_CARDS * cards_z
        - W_OPP * opp_z
    )

    raw_std = np.nanstd(delta_raw)
    if raw_std > 0:
        scale = TARGET_DELTA_STD / raw_std
    else:
        scale = 1.0

    df["delta_price"] = delta_raw * scale
    return df


# =============================================================================
# 3. APPLY RECENCY WEIGHTS + BUILD CUMULATIVE PRICE PATH
# =============================================================================

def apply_recency_and_cumulate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply season-based recency weights and build a cumulative price path.

    Recency weights:
        weight(season_year) = RECENCY_LAMBDA ** (max_year - season_year)

    Price process for each team:
        P_t = BASE_PRICE + Σ_{i ≤ t} weight(season_i) * ΔP_i
    """
    df = df.sort_values(["team", "season_year", "date"]).copy()

    if "delta_price" not in df.columns:
        raise ValueError("compute_delta_price must be called before this function.")

    max_year = df["season_year"].max()
    df["recency_weight"] = RECENCY_LAMBDA ** (max_year - df["season_year"])

    df["delta_price_weighted"] = df["delta_price"] * df["recency_weight"]

    price_list = []
    for team, grp in df.groupby("team", sort=False):
        grp = grp.sort_values(["season_year", "date"])
        # start at BASE_PRICE
        prices = [BASE_PRICE]
        for dp in grp["delta_price_weighted"].iloc[1:].values:
            prices.append(prices[-1] + dp)
        grp["price_raw"] = prices
        price_list.append(grp)

    out = pd.concat(price_list, ignore_index=True)
    return out


# =============================================================================
# 4. RESCALE FINAL 2025–2026 PRICES INTO [TARGET_MIN_PRICE, TARGET_MAX_PRICE]
# =============================================================================

def rescale_final_season(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rescale raw price paths so that final prices in FINAL_SEASON lie in
    [TARGET_MIN_PRICE, TARGET_MAX_PRICE].

    Steps:
        - Take the last match in FINAL_SEASON for each team,
        - Compute min and max of price_raw across those teams,
        - Linearly scale ALL price_raw so that:
              min_final → TARGET_MIN_PRICE
              max_final → TARGET_MAX_PRICE.
    """
    df = df.copy()

    if "price_raw" not in df.columns:
        raise ValueError("apply_recency_and_cumulate must be called before this function.")

    final_season_df = df[df["season"] == FINAL_SEASON].copy()
    if final_season_df.empty:
        raise ValueError(
            f"No rows found for FINAL_SEASON={FINAL_SEASON}. "
            "Check that this season exists in team_match_features.csv."
        )

    last_final = (
        final_season_df.sort_values("date")
        .groupby("team", as_index=False)
        .tail(1)
    )

    raw_min = last_final["price_raw"].min()
    raw_max = last_final["price_raw"].max()

    if np.isclose(raw_min, raw_max):
        df["price_scaled"] = TARGET_MIN_PRICE
        return df

    scale = (TARGET_MAX_PRICE - TARGET_MIN_PRICE) / (raw_max - raw_min)
    df["price_scaled"] = TARGET_MIN_PRICE + (df["price_raw"] - raw_min) * scale
    return df


# =============================================================================
# 5. VALIDATION REGRESSION (ΔPRICE ~ FACTORS)
# =============================================================================

def explanatory_regression(df: pd.DataFrame) -> Tuple[Optional[float], Optional[pd.Series]]:
    """
    Diagnostic regression:

        ΔPrice_t ~ pts_t + form3_t + xGD_t + card_points_t + opp_avg_pts_t

    This is *not* used for forecasting. It is only there to show in the report
    that the designed ΔPrice_t is indeed driven by the match-performance factors.

    Returns
    -------
    r2 : float or None
        R² of the regression (explanatory power).
    coefs : pd.Series or None
        Coefficients (on standardised features), indexed by feature name.
    """
    feature_cols = ["pts", "form3", "xGD", "card_points", "opp_avg_pts"]
    reg_df = df[feature_cols + ["delta_price"]].copy()
    reg_df = reg_df.replace([np.inf, -np.inf], np.nan).dropna()

    if len(reg_df) < 50:
        print("Not enough clean rows for regression; skipping diagnostic.")
        return None, None

    X = reg_df[feature_cols].values
    y = reg_df["delta_price"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)

    r2 = r2_score(y, y_pred)
    coefs = pd.Series(model.coef_, index=feature_cols)
    return r2, coefs


# =============================================================================
# 6. ORCHESTRATOR
# =============================================================================

def run_pricing_engine() -> pd.DataFrame:
    """
    Run the full pricing engine pipeline:

        1. Load engineered match features (2010+).
        2. Compute ΔPrice_t with manual, football-based weights.
        3. Apply season-based recency weights and cumulate to price paths.
        4. Rescale final 2025–2026 prices into [TARGET_MIN_PRICE, TARGET_MAX_PRICE].
        5. Run explanatory regression ΔPrice ~ factors as validation.
        6. Save time-series + final-season snapshot to CSV.
    """
    print(f"Loading match features from {MATCH_FILE} ...")
    df = load_match_features()

    print("Computing per-match ΔPrice using manual weights ...")
    df = compute_delta_price(df)

    print("Applying recency weights and building cumulative price paths ...")
    df = apply_recency_and_cumulate(df)

    print(
        f"Rescaling final {FINAL_SEASON} prices to "
        f"[{TARGET_MIN_PRICE}, {TARGET_MAX_PRICE}] ..."
    )
    df = rescale_final_season(df)

    print("Running diagnostic regression ΔPrice ~ performance factors ...")
    r2, coefs = explanatory_regression(df)
    if r2 is not None:
        print(f"\nExplanatory R² for ΔPrice model: {r2:.3f}")
        print("Regression coefficients (standardised features):")
        print(coefs.to_string())

    # Save full price time-series
    out_cols = [
        "date",
        "season",
        "season_year",
        "team",
        "result",
        "pts",
        "form3",
        "xGD",
        "card_points",
        "opp_avg_pts",
        "delta_price",
        "recency_weight",
        "delta_price_weighted",
        "price_raw",
        "price_scaled",
    ]
    keep_cols = [c for c in out_cols if c in df.columns]
    df[keep_cols].to_csv(TIMESERIES_OUT, index=False)
    print(f"\nSaved price time-series to: {TIMESERIES_OUT}")

    # Save final-season cross-section
    final = (
        df[df["season"] == FINAL_SEASON]
        .sort_values("date")
        .groupby("team", as_index=False)
        .tail(1)[["team", "price_scaled"]]
        .rename(columns={"price_scaled": "current_price"})
        .sort_values("current_price", ascending=False)
        .reset_index(drop=True)
    )

    final.to_csv(CURRENT_OUT, index=False)
    print(f"Saved current prices for {FINAL_SEASON} to: {CURRENT_OUT}")

    print(f"\nTop 5 teams by current price in {FINAL_SEASON}:")
    print(final.head(5).to_string(index=False))

    print(f"\nBottom 5 teams by current price in {FINAL_SEASON}:")
    print(final.tail(5).to_string(index=False))

    return df


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_pricing_engine()