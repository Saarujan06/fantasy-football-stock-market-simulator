from __future__ import annotations
from pathlib import Path
from typing import Optional  # Add this import

import numpy as np
import pandas as pd

# ---- Paths ----

# Project root = two levels up from this file
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Adjust these if your folder names are slightly different
CLEANED_DIR = PROJECT_ROOT / "data" / "Cleaned Data"

MATCHES_FILE = CLEANED_DIR / "Full_Prem_2020-2026_cleaned.csv"
XG_FILE = CLEANED_DIR / "cleaned_game_stats_2020onwards.csv"


def _load_matches() -> pd.DataFrame:
    """
    Load the full Premier League results file and return a match-level DataFrame.
    Expected columns in MATCHES_FILE:
        Date, Season (optional), HomeTeam, AwayTeam,
        FTHG, FTAG, FTR, HY, AY, HR, AR
    """
    df = pd.read_csv(MATCHES_FILE)

    # Normalise column names (in case of weird capitalisation)
    df.columns = [c.strip() for c in df.columns]

    # Parse date
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")  # Removed dayfirst=True
    df = df.dropna(subset=["Date"])

    # If there is no Season column, create one from the year
    if "Season" not in df.columns:
        # simple version: season == calendar year of start
        df["Season"] = df["Date"].dt.year

    return df


def _expand_home_away(df_matches: pd.DataFrame) -> pd.DataFrame:
    """
    From match-level rows, create team-level rows.

    Each match becomes two rows:
      - one for the home team
      - one for the away team

    Output columns:
      date, season, team, opponent, home_away,
      goals_for, goals_against, result, pts, yellow, red
    """
    # Home team rows
    home = pd.DataFrame(
        {
            "date": df_matches["Date"],
            "season": df_matches["Season"],
            "team": df_matches["HomeTeam"],
            "opponent": df_matches["AwayTeam"],
            "home_away": "h",  # Changed to lowercase for consistency
            "goals_for": df_matches["FTHG"],
            "goals_against": df_matches["FTAG"],
            "yellow": df_matches["HY"],
            "red": df_matches["HR"],
            "full_result": df_matches["FTR"],
        }
    )

    # Away team rows
    away = pd.DataFrame(
        {
            "date": df_matches["Date"],
            "season": df_matches["Season"],
            "team": df_matches["AwayTeam"],
            "opponent": df_matches["HomeTeam"],
            "home_away": "a",  # Changed to lowercase for consistency
            "goals_for": df_matches["FTAG"],
            "goals_against": df_matches["FTHG"],
            "yellow": df_matches["AY"],
            "red": df_matches["AR"],
            "full_result": df_matches["FTR"],
        }
    )

    team_df = pd.concat([home, away], ignore_index=True)

    # Convert FTR (H/D/A) to result from *this* team's perspective
    def _result_from_pov(row) -> str:
        if row["full_result"] == "D":
            return "D"
        if row["full_result"] == "H" and row["home_away"] == "h":
            return "W"
        if row["full_result"] == "A" and row["home_away"] == "a":
            return "W"
        return "L"

    team_df["result"] = team_df.apply(_result_from_pov, axis=1)

    # Points: W=3, D=1, L=0
    team_df["pts"] = team_df["result"].map({"W": 3, "D": 1, "L": 0}).astype(int)

    # Tidy up
    team_df = team_df.drop(columns=["full_result"])

    return team_df


def _load_xg() -> Optional[pd.DataFrame]:
    """
    Load Understat-style xG file and return EPL-only rows.

    Expected columns in XG_FILE (based on your screenshot):
        league, season, club_name, home_away, xG, xGA, date, ...
    We keep only what we need.
    """
    if not XG_FILE.exists():
        print(f"⚠️ xG file not found at {XG_FILE}. Skipping xG merge.")
        return None

    df = pd.read_csv(XG_FILE)

    # Normalise column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Filter to Premier League only
    if "league" in df.columns:
        df = df[df["league"] == "EPL"]

    # Parse date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
    else:
        raise ValueError("xG file must contain a 'date' column.")

    # Standardise names
    rename_map = {}
    if "club_name" in df.columns:
        rename_map["club_name"] = "team"
    if "xg" in df.columns:
        rename_map["xg"] = "xG_for"
    if "xga" in df.columns:
        rename_map["xga"] = "xG_against"

    df = df.rename(columns=rename_map)

    # Keep only useful columns
    keep_cols = ["date", "team"]
    for col in ["xG_for", "xG_against", "season", "home_away"]:
        if col in df.columns:
            keep_cols.append(col)

    df = df[keep_cols].copy()

    return df


def load_team_match_panel() -> pd.DataFrame:
    """
    Public function: returns the main panel with one row per team per match.

    Columns:
        date, season, team, opponent, home_away,
        goals_for, goals_against, result, pts, yellow, red,
        xG_for (optional), xG_against (optional)
    """
    # 1) Match results -> team-level
    matches = _load_matches()
    team_panel = _expand_home_away(matches)

    # 2) xG merge
    xg_df = _load_xg()
    if xg_df is not None:
        # Normalize team names for both dataframes
        xg_df["team"] = xg_df["team"].astype(str).str.strip()
        team_panel["team"] = team_panel["team"].astype(str).str.strip()

        # Merge on date, team, and home_away for more accurate matching
        merge_cols = ["date", "team"]
        if "home_away" in xg_df.columns:
            merge_cols.append("home_away")
        
        team_panel = team_panel.merge(
            xg_df,
            on=merge_cols,
            how="left",
            suffixes=("", "_xg")  # Add suffix to avoid column conflicts
        )
    else:
        print("⚠️ Skipping xG merge because xG data is not available.")

    # Sort for convenience
    team_panel = team_panel.sort_values(["team", "date"]).reset_index(drop=True)

    return team_panel


if __name__ == "__main__":
    # Small debug run: print shape & head when you execute:
    #   python3 -m src.calculations.load_data
    df_panel = load_team_match_panel()
    print(f"Shape: {df_panel.shape}")
    print(f"Columns: {df_panel.columns.tolist()}")
    print("\nFirst 10 rows:")
    print(df_panel.head(10))
    print(f"\nRows with xG data: {df_panel['xG_for'].notna().sum() if 'xG_for' in df_panel.columns else 0}")