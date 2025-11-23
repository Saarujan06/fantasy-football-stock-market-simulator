from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEAN_DIR = PROJECT_ROOT / "data" / "cleaned_data"

RAW_PANEL = CLEAN_DIR / "Full_Prem_2020-2026_cleaned.csv"
OUT_FILE = CLEAN_DIR / "team_match_features.csv"


# ---------------------------------------------------------
# Helper: Convert a single match into two team-rows
# ---------------------------------------------------------
def expand_match_row(row):
    """Convert 1 match into 2 rows: home team & away team."""
    date = row["date"]
    season = row["season"]

    # Home team row
    home = {
        "date": date,
        "season": season,
        "team": row["home_team"],
        "opponent": row["away_team"],
        "is_home": 1,

        "goals_for": row["goals_home"],
        "goals_against": row["goals_away"],
        "shots_for": row["hs"],
        "shots_against": row["as"],
        "shots_on_target_for": row["hst"],
        "shots_on_target_against": row["ast"],
        "fouls_for": row["hf"],
        "fouls_against": row["af"],
        "corners_for": row["hc"],
        "corners_against": row["ac"],
        "yellow": row["hy"],
        "red": row["hr"],
    }

    # Away team row
    away = {
        "date": date,
        "season": season,
        "team": row["away_team"],
        "opponent": row["home_team"],
        "is_home": 0,

        "goals_for": row["goals_away"],
        "goals_against": row["goals_home"],
        "shots_for": row["as"],
        "shots_against": row["hs"],
        "shots_on_target_for": row["ast"],
        "shots_on_target_against": row["hst"],
        "fouls_for": row["af"],
        "fouls_against": row["hf"],
        "corners_for": row["ac"],
        "corners_against": row["hc"],
        "yellow": row["ay"],
        "red": row["ar"],
    }

    return home, away


# ---------------------------------------------------------
# Feature builders
# ---------------------------------------------------------
def compute_points(df):
    """Convert match results into points (3/1/0)."""
    df["pts"] = np.where(
        df["goals_for"] > df["goals_against"], 3,
        np.where(df["goals_for"] == df["goals_against"], 1, 0)
    )
    return df


def compute_card_points(df):
    """Assign penalty score for cards."""
    df["card_points"] = df["yellow"] * 1 + df["red"] * 3
    return df


def compute_form(df):
    """Rolling form over last 3 matches."""
    df = df.sort_values(["team", "date"])
    df["form3"] = df.groupby("team")["pts"].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    return df


def compute_rolling_opponent_strength(df):
    """Rolling average of opponent points."""
    opp_points = df[["date", "team", "pts"]].rename(columns={"team": "opponent", "pts": "opp_pts"})
    merged = df.merge(opp_points, on=["opponent", "date"], how="left")
    merged["opp_avg_pts"] = merged.groupby("team")["opp_pts"].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    merged = merged.drop(columns=["opp_pts"])
    return merged


def compute_xg_diff(df):
    """Create xG differential placeholder until you supply real xG."""
    df["xg_for"] = df["shots_on_target_for"] * 0.25
    df["xg_against"] = df["shots_on_target_against"] * 0.25
    df["xGD"] = df["xg_for"] - df["xg_against"]
    return df


# ---------------------------------------------------------
# Main builder
# ---------------------------------------------------------
def build_feature_dataset(save_to_csv=True):
    if not RAW_PANEL.exists():
        raise FileNotFoundError(f"Missing cleaned match file: {RAW_PANEL}")

    print("STEP 1 – Loading raw match file...")
    df_raw = pd.read_csv(RAW_PANEL, parse_dates=["date"])

    required_cols = [
        "date", "season", "home_team", "away_team",
        "goals_home", "goals_away",
        "hs", "as", "hst", "ast",
        "hf", "af", "hc", "ac",
        "hy", "ay", "hr", "ar"
    ]
    missing = [c for c in required_cols if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Missing columns in cleaned match file: {missing}")

    # Expand into team-match long format
    print("STEP 2 – Expanding matches into team-based rows...")
    rows = []
    for _, row in df_raw.iterrows():
        h, a = expand_match_row(row)
        rows.append(h)
        rows.append(a)

    df = pd.DataFrame(rows)

    # Compute features
    print("STEP 3 – Computing match features...")

    df = compute_points(df)
    df = compute_card_points(df)
    df = compute_form(df)
    df = compute_rolling_opponent_strength(df)
    df = compute_xg_diff(df)

    df = df.sort_values(["team", "date"]).reset_index(drop=True)

    if save_to_csv:
        df.to_csv(OUT_FILE, index=False)
        print(f"\n✅ Saved feature dataset → {OUT_FILE}")

    print("\nSample rows:")
    print(df.head())

    return df


if __name__ == "__main__":
    build_feature_dataset(save_to_csv=True)