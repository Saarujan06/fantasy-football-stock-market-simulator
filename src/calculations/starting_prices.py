from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import numpy as np


# -----------------------------------------------------
# Paths and constants
# -----------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEAN_DIR = PROJECT_ROOT / "data" / "cleaned_data"

FEATURES_FILE = CLEAN_DIR / "team_match_features.csv"
STARTING_PRICES_OUT = CLEAN_DIR / "starting_prices.csv"

# Season we treat as "current"
TARGET_SEASON = "2025-2026"

# Current 2025/26 Premier League teams
CURRENT_TEAMS = [
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford",
    "Brighton and Hove Albion", "Burnley", "Chelsea", "Crystal Palace",
    "Everton", "Fulham", "Leeds United", "Liverpool",
    "Manchester City", "Manchester United", "Newcastle United",
    "Nottingham Forest", "Sunderland", "Tottenham Hotspur",
    "West Ham United", "Wolverhampton Wanderers",
]


# -----------------------------------------------------
# Core logic
# -----------------------------------------------------
def build_starting_prices(
    price_min: float = 80.0,
    price_max: float = 120.0,
) -> pd.DataFrame:
    """
    Build starting prices as an accumulated value of pre-2025 seasons.

    Logic:
      - Load team_match_features.csv
      - Filter to all seasons BEFORE TARGET_SEASON (i.e. "2019–2024 history")
      - For each team: sum total points over those seasons
      - Linearly map total points into [price_min, price_max]
      - For CURRENT_TEAMS with NO history, assign a price below the minimum

    Returns:
      DataFrame with columns: team, hist_total_pts, start_price
    """
    if not FEATURES_FILE.exists():
        raise FileNotFoundError(f"Features file not found: {FEATURES_FILE}")

    df = pd.read_csv(FEATURES_FILE, parse_dates=["date"])

    if "season" not in df.columns:
        raise ValueError(
            "Expected a 'season' column in team_match_features.csv "
            " (e.g. '2020-2021', '2021-2022', ...)"
        )

    if "pts" not in df.columns:
        raise ValueError("Expected a 'pts' column in team_match_features.csv")

    # -------------------------------------------------
    # 1. Filter to historical seasons (2019–2024)
    #    Here: all seasons != TARGET_SEASON
    # -------------------------------------------------
    hist = df[df["season"] != TARGET_SEASON].copy()
    if hist.empty:
        raise ValueError(
            "No historical seasons found (all rows are TARGET_SEASON). "
            "You need multiple seasons in team_match_features.csv."
        )

    # Ensure we only consider CURRENT_TEAMS (nice for your context)
    hist = hist[hist["team"].isin(CURRENT_TEAMS)].copy()

    # -------------------------------------------------
    # 2. Aggregate total points per team across history
    # -------------------------------------------------
    hist["pts"] = hist["pts"].fillna(0.0)
    hist_total = (
        hist.groupby("team")["pts"]
        .sum()
        .rename("hist_total_pts")
        .to_frame()
    )

    if hist_total.empty:
        raise ValueError(
            "Historical aggregation is empty – check that team names in "
            "CURRENT_TEAMS match the 'team' column exactly."
        )

    # -------------------------------------------------
    # 3. Map total points into a price range [price_min, price_max]
    # -------------------------------------------------
    pts_min = float(hist_total["hist_total_pts"].min())
    pts_max = float(hist_total["hist_total_pts"].max())

    if np.isclose(pts_max, pts_min):
        # Degenerate case: all teams have the same total points
        hist_total["start_price"] = (price_min + price_max) / 2.0
    else:
        # Normalised score in [0,1]
        hist_total["score_norm"] = (
            (hist_total["hist_total_pts"] - pts_min) / (pts_max - pts_min)
        )
        hist_total["start_price"] = (
            price_min + hist_total["score_norm"] * (price_max - price_min)
        )

    # -------------------------------------------------
    # 4. Ensure all CURRENT_TEAMS are present
    #    For teams with NO historical data, assign a low price
    # -------------------------------------------------
    df_start = hist_total.copy()

    # If some CURRENT_TEAMS are missing (e.g. true IPOs like Sunderland),
    # we give them a price below the historical minimum.
    existing_teams = set(df_start.index.tolist())
    missing_teams = [t for t in CURRENT_TEAMS if t not in existing_teams]

    if not df_start.empty:
        min_start_price = float(df_start["start_price"].min())
    else:
        min_start_price = price_min

    ipo_price = min_start_price - 5.0  # "below the lowest finishing team"

    ipo_rows = []
    for team in missing_teams:
        ipo_rows.append(
            {
                "team": team,
                "hist_total_pts": 0.0,
                "start_price": ipo_price,
            }
        )

    if ipo_rows:
        ipo_df = pd.DataFrame(ipo_rows).set_index("team")
        df_start = pd.concat([df_start, ipo_df], axis=0)

    # Reset index to have 'team' as a column
    df_start = df_start.reset_index()

    # Clean up helper column if present
    if "score_norm" in df_start.columns:
        df_start = df_start.drop(columns=["score_norm"])

    # -------------------------------------------------
    # 5. Save and print summary
    # -------------------------------------------------
    STARTING_PRICES_OUT.parent.mkdir(parents=True, exist_ok=True)
    df_start.to_csv(STARTING_PRICES_OUT, index=False)

    print(f"\n✅ Starting prices written to: {STARTING_PRICES_OUT}\n")

    print("=== Starting price summary (2019–2024 accumulated points) ===")
    print(df_start.sort_values("start_price", ascending=False).head(10))
    print("\nLowest 5 start prices:")
    print(df_start.sort_values("start_price", ascending=True).head(5))

    return df_start


# -----------------------------------------------------
# CLI entry point
# -----------------------------------------------------
if __name__ == "__main__":
    build_starting_prices()