from __future__ import annotations
from pathlib import Path

import pandas as pd
from .load_data import load_team_match_panel

# Where to save output
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_OUT = PROJECT_ROOT / "data" / "Cleaned Data" / "team_match_features.csv"

# ----------------------------------------------
# Premier League 2025/2026 Teams (STATIC FILTER)
# ----------------------------------------------
CURRENT_TEAMS = [
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton and Hove Albion", 
    "Burnley", "Chelsea", "Crystal Palace", "Everton", "Fulham", "Leeds United", 
    "Liverpool", "Manchester City", "Manchester United", "Newcastle United", 
    "Nottingham Forest", "Sunderland", "Tottenham Hotspur", "West Ham United", 
    "Wolverhampton Wanderers"
]


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to the team-match panel"""
    df = df.sort_values(["team", "date"]).copy()

    # ---- xG difference ----
    if "xG_for" not in df.columns:
        df["xG_for"] = 0.0
    if "xG_against" not in df.columns:
        df["xG_against"] = 0.0

    df["xG_for"] = df["xG_for"].fillna(0.0)
    df["xG_against"] = df["xG_against"].fillna(0.0)
    df["xGD"] = df["xG_for"] - df["xG_against"]

    # ---- clean sheet ----
    df["clean_sheet"] = (df["goals_against"] == 0).astype(int)

    # ---- card points ----
    df["yellow"] = df["yellow"].fillna(0).astype(int)
    df["red"] = df["red"].fillna(0).astype(int)
    df["card_points"] = df["yellow"] + 2 * df["red"]

    # ---- rolling 3-match form ----
    df["form3"] = (
        df.groupby("team")["pts"]
          .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    )

    # ---- cumulative team average pts ----
    df["team_avg_pts"] = (
        df.groupby("team")["pts"]
          .transform(lambda x: x.expanding().mean())
    )

    # ---- opponent strength ----
    opp_avg = (
        df[["date", "team", "team_avg_pts"]]
        .rename(columns={"team": "opponent", "team_avg_pts": "opp_avg_pts"})
    )

    df = df.merge(
        opp_avg,
        on=["date", "opponent"],
        how="left",
        suffixes=("", "_drop")
    )

    league_mean = df["team_avg_pts"].mean()
    df["opp_avg_pts"] = df["opp_avg_pts"].fillna(league_mean)

    # Drop any duplicate columns from merge
    df = df.loc[:, ~df.columns.str.endswith("_drop")]

    return df


def build_feature_dataset() -> pd.DataFrame:
    """
    Main pipeline:
    1. Load team-match panel
    2. Filter to current Premier League teams
    3. Add engineered features
    4. Save to CSV (only if doesn't exist)
    """
    print("Loading team-match panel...")
    base = load_team_match_panel()
    
    print(f"Initial dataset: {len(base)} rows, {len(base['team'].unique())} unique teams")

    # -------------------------------------------------------
    # Filter: KEEP ONLY 2025/26 Premier League clubs
    # -------------------------------------------------------
    print("\nFiltering to current 2025/26 Premier League teams...")
    base = base[base["team"].isin(CURRENT_TEAMS)].copy()

    teams_found = sorted(base["team"].unique())
    print(f"Teams included ({len(teams_found)}): {teams_found}")

    if len(teams_found) < len(CURRENT_TEAMS):
        missing = set(CURRENT_TEAMS) - set(teams_found)
        print(f"\n⚠️ WARNING: Missing teams: {sorted(missing)}")
        print("Check team name capitalization in your dataset!")

    # -------------------------------------------------------
    # Add computed features
    # -------------------------------------------------------
    print("\nAdding engineered features...")
    features = add_basic_features(base)

    # -------------------------------------------------------
    # Save to CSV (only if file does NOT exist)
    # -------------------------------------------------------
    FEATURES_OUT.parent.mkdir(parents=True, exist_ok=True)

    if not FEATURES_OUT.exists():
        features.to_csv(FEATURES_OUT, index=False)
        print(f"✅ Created feature dataset: {FEATURES_OUT}")
    else:
        print(f"📁 Feature dataset already exists: {FEATURES_OUT}")

    print(f"\nFinal dataset shape: {features.shape}")
    print(f"Columns: {features.columns.tolist()}")
    print(f"Date range: {features['date'].min()} → {features['date'].max()}")
    print(f"Rows with xG data: {features['xG_for'].gt(0).sum()} / {len(features)}")

    return features


if __name__ == "__main__":
    df_feat = build_feature_dataset()
    print("\n===== SAMPLE DATA =====")
    print(df_feat[["date", "team", "opponent", "pts", "xGD", "clean_sheet", "card_points", "form3", "opp_avg_pts"]].head(20))
    print("\n===== SUMMARY STATISTICS =====")
    print(df_feat[["pts", "xGD", "clean_sheet", "card_points", "form3", "opp_avg_pts"]].describe())