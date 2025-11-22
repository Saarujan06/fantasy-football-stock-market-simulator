from __future__ import annotations
from pathlib import Path

import pandas as pd
from .load_data import load_team_match_panel

# Where to save output
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_OUT = PROJECT_ROOT / "data" / "Cleaned Data" / "team_match_features.csv"


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given the team-match panel from load_team_match_panel(), add:
      - xGD: xG_for - xG_against
      - clean_sheet: 1 if goals_against == 0 else 0
      - card_points: yellow + 2 * red
      - form3: rolling 3-match average of pts for each team
      - team_avg_pts: cumulative average pts per team
      - opp_avg_pts: opponent's cumulative average pts (opponent strength)
    """
    df = df.sort_values(["team", "date"]).copy()

    # ---- xG difference ----
    if "xG_for" not in df.columns:
        df["xG_for"] = 0.0
    if "xG_against" not in df.columns:
        df["xG_against"] = 0.0

    df["xG_for"] = df["xG_for"].fillna(0.0)
    df["xG_against"] = df["xG_against"].fillna(0.0)
    df["xGD"] = df["xG_for"] - df["xG_against"]

    # ---- clean sheet indicator ----
    if "goals_against" not in df.columns:
        raise ValueError("Column 'goals_against' missing from team-match panel.")
    df["clean_sheet"] = (df["goals_against"] == 0).astype(int)

    # ---- card points ----
    df["yellow"] = df.get("yellow", 0).fillna(0).astype(int)
    df["red"] = df.get("red", 0).fillna(0).astype(int)
    df["card_points"] = df["yellow"] + 2 * df["red"]

    # ---- rolling 3-match form (average pts over last 3 matches) ----
    if "pts" not in df.columns:
        raise ValueError("Column 'pts' missing from team-match panel.")
    df["form3"] = (
        df.groupby("team")["pts"]
          .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    )

    # ---- team cumulative average points (for opponent strength) ----
    df["team_avg_pts"] = (
        df.groupby("team")["pts"]
          .transform(lambda x: x.expanding().mean())
    )

    # ---- opponent strength: opponent's avg pts up to that date ----
    opp_avg = (
        df[["date", "team", "team_avg_pts"]]
        .rename(columns={"team": "opponent", "team_avg_pts": "opp_avg_pts"})
    )

    df = df.merge(
        opp_avg,
        on=["date", "opponent"],
        how="left",
        suffixes=("", "_drop"),
    )

    league_mean = df["team_avg_pts"].mean()
    df["opp_avg_pts"] = df["opp_avg_pts"].fillna(league_mean)

    # Drop duplicate columns created by merge
    df = df.loc[:, ~df.columns.str.endswith("_drop")]

    return df


def build_feature_dataset() -> pd.DataFrame:
    """
    Main pipeline:
    1. Load full team-match panel (ALL teams, ALL seasons in data)
    2. Add engineered features
    3. Save to CSV
    """
    print("Loading team-match panel (ALL teams)...")
    base = load_team_match_panel()
    print(f"Initial dataset: {len(base)} rows, {len(base['team'].unique())} unique teams")

    print("\nAdding engineered features...")
    features = add_basic_features(base)

    FEATURES_OUT.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(FEATURES_OUT, index=False)

    print(f"\n✅ Saved feature dataset to: {FEATURES_OUT}")
    print(f"Final dataset shape: {features.shape}")
    print(f"Date range: {features['date'].min()} → {features['date'].max()}")
    print(f"Sample teams: {sorted(features['team'].unique())[:10]} ...")

    return features


if __name__ == "__main__":
    df_feat = build_feature_dataset()
    print("\n===== SAMPLE DATA =====")
    print(
        df_feat[
            ["date", "team", "opponent", "pts", "xGD",
             "clean_sheet", "card_points", "form3", "opp_avg_pts"]
        ].head(20)
    )