from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from src.utils.team_names import normalize_team_name, CANONICAL_TEAMS_2526

# =============================================================================
# PATHS & GLOBAL CONFIG
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "cleaned_data"
MATCH_FILE = DATA_DIR / "team_match_features.csv"

RESULTS_DIR = PROJECT_ROOT / "results" / "pricing_engine"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TIMESERIES_OUT = RESULTS_DIR / "price_timeseries.csv"
CURRENT_OUT = RESULTS_DIR / "current_prices_2025_2026.csv"

HISTORY_START_YEAR = 2010
FINAL_SEASON = "2025_2026"

BASE_PRICE = 100.0
TARGET_DELTA_STD = 2.0
RECENCY_LAMBDA = 0.92

TARGET_MIN_PRICE = 50.0
TARGET_MAX_PRICE = 1000.0

W_PTS = 1.0
W_FORM = 0.8
W_XGD = 0.7
W_CARDS = 0.3
W_OPP = 0.5


# =============================================================================
# Helpers
# =============================================================================

def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def _require_any(df: pd.DataFrame, candidates: list[str], msg: str) -> str:
    col = _find_col(df, candidates)
    if col is None:
        raise KeyError(msg + f" Tried: {candidates}. Found: {list(df.columns)}")
    return col


# =============================================================================
# 1. LOAD MATCH FEATURES (robust to your build_features output)
# =============================================================================

def load_match_features() -> pd.DataFrame:
    if not MATCH_FILE.exists():
        raise FileNotFoundError(
            f"{MATCH_FILE} not found.\n"
            "Run:\n"
            "  python -m src.pipeline.build_features"
        )

    df = pd.read_csv(MATCH_FILE)
    df.columns = [c.strip() for c in df.columns]

    # required basics
    season_col = _require_any(df, ["season"], "Expected 'season' column in team_match_features.csv.")
    date_col = _require_any(df, ["date"], "Expected 'date' column in team_match_features.csv.")
    team_col = _require_any(df, ["team"], "Expected 'team' column in team_match_features.csv.")

    # normalize team names
    df[team_col] = df[team_col].apply(normalize_team_name)

    # season_year
    df["season_year"] = df[season_col].astype(str).str.split("_").str[0].astype("Int64")
    df = df[df["season_year"] >= HISTORY_START_YEAR].copy()

    # date
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).copy()

    # --- Build pts (3/1/0) ---
    # Your build_features uses label = result (0=loss,1=draw,2=win)
    label_col = _find_col(df, ["label", "result", "outcome"])
    if label_col is None:
        raise KeyError("Need a match outcome column (e.g. 'label' with 0/1/2).")

    # If it's strings like W/D/L, map it. If numeric 0/1/2, use directly.
    if df[label_col].dtype == object:
        s = df[label_col].astype(str).str.upper().str.strip()
        # accept W/D/L or H/D/A style if present
        mapping = {"W": 2, "D": 1, "L": 0, "H": 2, "A": 0}
        df["label_num"] = s.map(mapping)
    else:
        df["label_num"] = pd.to_numeric(df[label_col], errors="coerce")

    df = df.dropna(subset=["label_num"]).copy()
    df["label_num"] = df["label_num"].astype(int)

    df["pts"] = df["label_num"].map({0: 0, 1: 1, 2: 3}).astype(float)

    # --- Build form3 ---
    # Prefer a precomputed rolling points column if available (points_last3)
    points_last3 = _find_col(df, ["points_last3", "pts_last3", "points_rolling3"])
    if points_last3 is not None:
        df["form3"] = pd.to_numeric(df[points_last3], errors="coerce")
    else:
        # compute ourselves, leakage-safe (past only)
        df = df.sort_values([team_col, "season_year", date_col]).copy()
        df["form3"] = (
            df.groupby(team_col)["pts"]
              .apply(lambda s: s.shift(1).rolling(window=3, min_periods=1).mean())
              .reset_index(level=0, drop=True)
        )

    # --- Build xGD ---
    # Prefer xG_last3 - xGA_last3 if available
    xg3 = _find_col(df, ["xg_last3", "xG_last3", "xG_for_last3"])
    xga3 = _find_col(df, ["xga_last3", "xGA_last3", "xG_against_last3"])
    if xg3 is not None and xga3 is not None:
        df["xGD"] = pd.to_numeric(df[xg3], errors="coerce") - pd.to_numeric(df[xga3], errors="coerce")
    else:
        # fallback: npxGD if present, else 0 (but we will assert we have some signal later)
        npxgd = _find_col(df, ["npxgd", "npxGD"])
        if npxgd is not None:
            df["xGD"] = pd.to_numeric(df[npxgd], errors="coerce")
        else:
            df["xGD"] = np.nan

    # --- card_points + opp_avg_pts (optional; if missing we set to 0) ---
    cards_col = _find_col(df, ["card_points", "cards", "card_pts"])
    opp_col = _find_col(df, ["opp_avg_pts", "opp_avg_points", "opponent_avg_pts", "opponent_avg_points"])

    df["card_points"] = pd.to_numeric(df[cards_col], errors="coerce") if cards_col else 0.0
    df["opp_avg_pts"] = pd.to_numeric(df[opp_col], errors="coerce") if opp_col else 0.0

    # unify column names we use downstream
    df = df.rename(columns={season_col: "season", date_col: "date", team_col: "team"}).copy()

    # numeric cleanup
    for c in ["pts", "form3", "xGD", "card_points", "opp_avg_pts"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values(["team", "season_year", "date"]).reset_index(drop=True)

    # IMPORTANT: do not allow "everything becomes 0" silently
    # If these are all NaN/0, pricing cannot work.
    signal_sum = (
        df["pts"].fillna(0).abs().sum()
        + df["form3"].fillna(0).abs().sum()
        + df["xGD"].fillna(0).abs().sum()
    )
    if signal_sum == 0:
        raise ValueError(
            "Pricing inputs have no signal (pts/form3/xGD all zero). "
            "This means pricing_engine is not aligned with team_match_features columns."
        )

    # Fill remaining NaNs (xGD can be NaN pre-Understat), but AFTER the signal check
    df[["pts", "form3", "xGD", "card_points", "opp_avg_pts"]] = df[
        ["pts", "form3", "xGD", "card_points", "opp_avg_pts"]
    ].fillna(0.0)

    return df


# =============================================================================
# 2. MANUAL ΔPRICE FORMULA
# =============================================================================

def compute_delta_price(df: pd.DataFrame) -> pd.DataFrame:
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
    scale = TARGET_DELTA_STD / raw_std if raw_std > 0 else 1.0

    df["delta_price"] = delta_raw * scale
    return df


# =============================================================================
# 3. APPLY RECENCY WEIGHTS + CUMULATE
# =============================================================================

def apply_recency_and_cumulate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["team", "season_year", "date"]).copy()

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

    return pd.concat(price_list, ignore_index=True)


# =============================================================================
# 4. RESCALE FINAL SEASON
# =============================================================================

def rescale_final_season(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    final_season_df = df[df["season"] == FINAL_SEASON].copy()
    if final_season_df.empty:
        raise ValueError(
            f"No rows found for FINAL_SEASON={FINAL_SEASON}. "
            "Your team_match_features.csv likely doesn't include 2025_2026 yet."
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
# 5. DIAGNOSTIC REGRESSION
# =============================================================================

def explanatory_regression(df: pd.DataFrame) -> Tuple[Optional[float], Optional[pd.Series]]:
    feature_cols = ["pts", "form3", "xGD", "card_points", "opp_avg_pts"]
    reg_df = df[feature_cols + ["delta_price"]].copy()
    reg_df = reg_df.replace([np.inf, -np.inf], np.nan).dropna()

    if len(reg_df) < 50:
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
    print(f"Loading match features from {MATCH_FILE} ...")
    df = load_match_features()

    print("Computing per-match ΔPrice using manual weights ...")
    df = compute_delta_price(df)

    print("Applying recency weights and building cumulative price paths ...")
    df = apply_recency_and_cumulate(df)

    print(f"Rescaling final {FINAL_SEASON} prices to [{TARGET_MIN_PRICE}, {TARGET_MAX_PRICE}] ...")
    df = rescale_final_season(df)

    r2, coefs = explanatory_regression(df)
    if r2 is not None:
        print(f"Explanatory R² for ΔPrice model: {r2:.3f}")
        print(coefs.to_string())

    # Save full time-series
    out_cols = [
        "date", "season", "season_year", "team",
        "pts", "form3", "xGD", "card_points", "opp_avg_pts",
        "delta_price", "recency_weight", "delta_price_weighted",
        "price_raw", "price_scaled",
    ]
    df[out_cols].to_csv(TIMESERIES_OUT, index=False)
    print(f"Saved price time-series to: {TIMESERIES_OUT}")

    # Save final-season snapshot (ONLY current PL teams)
    final = (
        df[df["season"] == FINAL_SEASON]
        .sort_values("date")
        .groupby("team", as_index=False)
        .tail(1)[["team", "price_scaled"]]
        .rename(columns={"price_scaled": "current_price"})
    )
    final["team"] = final["team"].apply(normalize_team_name)
    final = final[final["team"].isin(CANONICAL_TEAMS_2526)].copy()

    final = final.sort_values("current_price", ascending=False).reset_index(drop=True)
    final.to_csv(CURRENT_OUT, index=False)
    print(f"Saved current prices for {FINAL_SEASON} to: {CURRENT_OUT}")

    print("\nTop 5 teams by current price:")
    print(final.head(5).to_string(index=False))

    return df


if __name__ == "__main__":
    run_pricing_engine()