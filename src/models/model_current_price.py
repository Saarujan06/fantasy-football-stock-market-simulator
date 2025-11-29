from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEAN_DIR = PROJECT_ROOT / "data" / "cleaned_data"

FEATURES_FILE = CLEAN_DIR / "team_match_features.csv"
START_PRICES_FILE = CLEAN_DIR / "starting_prices_2019.csv"
PRICES_OUT = CLEAN_DIR / "team_prices_current.csv"


# ---------------------------------------------------------
# 1. Load team–match features
# ---------------------------------------------------------
def load_features() -> pd.DataFrame:
    """
    Load the team–match feature panel.

    Expected columns (you told me these exist):

        date, season, team, opponent, is_home,
        goals_for, goals_against,
        shots_for, shots_against,
        shots_on_target_for, shots_on_target_against,
        fouls_for, fouls_against,
        corners_for, corners_against,
        yellow, red,
        pts, card_points,
        form3, opp_avg_pts,
        xg_for, xg_against, xGD
    """
    if not FEATURES_FILE.exists():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURES_FILE}\n"
            "Run data_loader.py first to build team_match_features.csv."
        )

    df = pd.read_csv(FEATURES_FILE, parse_dates=["date"])

    required = [
        "date",
        "season",
        "team",
        "is_home",
        "pts",
        "card_points",
        "form3",
        "opp_avg_pts",
        "xGD",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in feature file: {missing}")

    df = df.sort_values(["team", "date"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------
# 2. Starting-price map
# ---------------------------------------------------------
def load_start_prices(teams: pd.Index) -> Dict[str, float]:
    """
    Map each team -> starting stock price.

    If starting_prices_2019.csv exists, use it (columns: team, start_price).
    Otherwise default to 100 for all teams.
    """
    default_price = 100.0
    start_map: Dict[str, float] = {t: default_price for t in teams}

    if not START_PRICES_FILE.exists():
        print(
            f"⚠️ No starting price file found at {START_PRICES_FILE}. "
            f"Using default start price = {default_price} for all teams."
        )
        return start_map

    print(f"Loading starting prices from {START_PRICES_FILE} ...")
    sp = pd.read_csv(START_PRICES_FILE)

    if "team" not in sp.columns or "start_price" not in sp.columns:
        raise ValueError(
            f"{START_PRICES_FILE} must contain columns: 'team', 'start_price'"
        )

    for _, row in sp.iterrows():
        team = str(row["team"])
        price = float(row["start_price"])
        start_map[team] = price

    print("Sample starting prices:")
    print(sp.head())
    return start_map


# ---------------------------------------------------------
# 3. ΔP_t rule (match-level price change)
# ---------------------------------------------------------
def compute_delta_P(df: pd.DataFrame) -> pd.Series:
    """
    Compute per-match price change ΔP_t using a linear scoring rule:

        pts_dev      = pts   − avg_pts
        form_dev     = form3 − avg_form

        ΔP_t = k_pts  * pts_dev
             + k_form * form_dev
             + k_xgd  * xGD
             - k_cards* card_points
             - k_opp  * opp_avg_pts

    Then scaled by a global factor to keep prices in a sensible range.
    """
    df = df.copy()

    for col in ["pts", "form3", "xGD", "card_points", "opp_avg_pts"]:
        df[col] = df.get(col, 0.0).fillna(0.0)

    avg_pts = df["pts"].mean()
    avg_form = df["form3"].mean()

    pts_dev = df["pts"] - avg_pts
    form_dev = df["form3"] - avg_form

    # Hyperparameters (you can discuss / tweak these in the report)
    k_pts = 3.0
    k_form = 1.5
    k_xgd = 0.8
    k_cards = 0.2
    k_opp = 0.4

    delta_P = (
        k_pts * pts_dev
        + k_form * form_dev
        + k_xgd * df["xGD"]
        - k_cards * df["card_points"]
        - k_opp * df["opp_avg_pts"]
    )

    SCALE = 0.3
    return delta_P * SCALE


# ---------------------------------------------------------
# 4. Build full price time series (Model A engine)
# ---------------------------------------------------------
def build_current_price_dataset(save_to_csv: bool = True) -> pd.DataFrame:
    """
    Main pricing engine:

      1. Load team_match_features.csv
      2. Compute ΔP_t for each match
      3. Initialise each team at its starting price
      4. Build cumulative prices P_t = P_0 + Σ ΔP_τ
      5. Optionally save to team_prices_current.csv
    """
    print("Loading feature dataset...")
    df = load_features()
    print(f"Rows: {len(df)}, teams: {df['team'].nunique()}")

    # Per-match price change
    print("Computing per-match price changes ΔP_t ...")
    df["delta_P"] = compute_delta_P(df)

    # Starting prices
    unique_teams = df["team"].unique()
    start_map = load_start_prices(unique_teams)

    # Build cumulative prices per team
    def _price_one_team(group: pd.DataFrame) -> pd.DataFrame:
        t = group["team"].iloc[0]
        P0 = float(start_map.get(t, 100.0))

        group = group.sort_values("date").copy()
        group["price"] = P0 + group["delta_P"].cumsum()
        return group

    priced = df.groupby("team", group_keys=False).apply(_price_one_team)

    # Avoid negative prices
    priced["price"] = priced["price"].clip(lower=1.0)

    if save_to_csv:
        PRICES_OUT.parent.mkdir(parents=True, exist_ok=True)
        priced.to_csv(PRICES_OUT, index=False)
        print(f"\n✅ Saved current price dataset to: {PRICES_OUT}")

    # Quick price summary
    final = (
        priced.sort_values("date")
        .groupby("team")["price"]
        .last()
        .sort_values(ascending=False)
    )

    print("\n=== CURRENT PRICE SUMMARY ===")
    print(f"Teams: {len(final)}")
    print(f"Price range: £{final.min():.2f} – £{final.max():.2f}")
    print("\nTop 5 teams by final price:")
    print(final.head())
    print("\nBottom 5 teams by final price:")
    print(final.tail())

    return priced


# ---------------------------------------------------------
# 5. History-based regression: price ~ cumulative history
# ---------------------------------------------------------
def evaluate_price_regression_with_history(priced: pd.DataFrame) -> None:
    """
    Regression of price on **historical / cumulative** features.

    For each team (sorted by date) we build:

        games_played   = 1, 2, 3, ...
        cum_pts        = cumulative sum of pts
        cum_xGD        = cumulative sum of xGD
        cum_cards      = cumulative sum of card_points

    We then fit a linear regression:

    Train/test split:
        - Train on seasons 2020–2021 .. 2023–2024
        - Test  on seasons 2024–2025 .. 2025–2026
    """

    df = priced.copy().sort_values(["team", "date"])

    # Build cumulative history features per team
    df["games_played"] = df.groupby("team").cumcount() + 1
    df["cum_pts"] = df.groupby("team")["pts"].cumsum()
    df["cum_xGD"] = df.groupby("team")["xGD"].cumsum()
    df["cum_cards"] = df.groupby("team")["card_points"].cumsum()

    # Numerically encode season by its starting year, e.g. '2020-2021' -> 2020
    season_start_year = df["season"].astype(str).str.slice(0, 4).astype(int)
    df["season_start"] = season_start_year

    feature_cols = [
        "games_played",
        "cum_pts",
        "cum_xGD",
        "cum_cards",
        "season_start",
    ]

    # Drop any rows with missing values (should be rare)
    df = df.dropna(subset=feature_cols + ["price"])

    X = df[feature_cols].values
    y = df["price"].values

    # Time-respecting split: train on seasons <= 2024 (i.e. up to 2024–2025)
    train_mask = df["season_start"] <= 2024
    test_mask = ~train_mask

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print("\n=== History-based regression of price (Model A evaluation) ===")
    print(f"Train size: {len(y_train)}, Test size: {len(y_test)}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    reg = LinearRegression()
    reg.fit(X_train_scaled, y_train)

    y_pred = reg.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    corr = np.corrcoef(y_test, y_pred)[0, 1]

    print(f"R² on held-out seasons: {r2:6.3f}")
    print(f"Correlation r(y, ŷ):   {corr:6.3f}\n")

    # Coefficients (after scaling)
    print("Regression coefficients (after scaling):")
    for name, beta in zip(feature_cols, reg.coef_):
        print(f"  {name:12s}: {beta:8.3f}")


# ---------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    # 1) build the full price panel using the pricing rule
    prices = build_current_price_dataset(save_to_csv=True)

    # 2) evaluate a history-based regression of price on cumulative features
    evaluate_price_regression_with_history(prices)

    # 3) show a few sample rows
    print("\nSample rows from priced dataset:")
    print(
        prices[
            [
                "date",
                "season",
                "team",
                "pts",
                "xGD",
                "form3",
                "card_points",
                "price",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )