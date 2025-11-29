from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# -------------------------------------------------------------------
# Paths and global constants
# -------------------------------------------------------------------

# data_loader.py now sits directly inside src/, so the project root is
# one level above this file.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
CLEAN_DIR = PROJECT_ROOT / "data" / "cleaned_data"

CLEAN_PANEL_FILE = CLEAN_DIR / "Full_Prem_2020-2026_cleaned.csv"
FEATURES_FILE = CLEAN_DIR / "team_match_features.csv"
SEASON_SUMMARY_FILE = CLEAN_DIR / "team_season_summary.csv"

# Mapping of seasons to raw CSV file names in data/raw
SEASONS_TO_FILES: Dict[str, str] = {
    "2020-2021": "prem_2020_2021.csv",
    "2021-2022": "prem_2021_2022.csv",
    "2022-2023": "prem_2022_2023.csv",
    "2023-2024": "prem_2023_2024.csv",
    "2024-2025": "prem_2024_2025.csv",
    "2025-2026": "prem_2025_2026.csv",
}


# -------------------------------------------------------------------
# 1) Loading and cleaning the raw match files
# -------------------------------------------------------------------

def _load_one_season(season: str, filename: str) -> pd.DataFrame:
    """
    Load and clean a single Premier League season from data/raw.

    The raw CSV is expected to follow the football-data.co.uk format with
    columns such as:
      Date, HomeTeam, AwayTeam,
      FTHG, FTAG, FTR,
      HS, AS, HST, AST,
      HF, AF, HC, AC,
      HY, AY, HR, AR

    Returns a DataFrame with:
      - parsed date column
      - snake_case column names
      - added 'season' column
    """
    path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Raw file not found for season {season}: {path}"
        )

    print(f"  Loading raw file for {season}: {path}")

    # football-data files are usually latin1 encoded
    df = pd.read_csv(path, encoding="latin1")

    required_cols = [
        "Date",
        "HomeTeam", "AwayTeam",
        "FTHG", "FTAG", "FTR",
        "HS", "AS", "HST", "AST",
        "HF", "AF", "HC", "AC",
        "HY", "AY", "HR", "AR",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"File {path} (season {season}) is missing columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    df = df[required_cols].copy()

    # Parse dates (day-first format in football-data)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    before = len(df)
    df = df.dropna(subset=["Date"])
    after = len(df)
    if after < before:
        print(f"    Dropped {before - after} rows with invalid dates in {season}")

    # Rename to nicer snake_case names
    df = df.rename(
        columns={
            "Date": "date",
            "HomeTeam": "home_team",
            "AwayTeam": "away_team",
            "FTHG": "goals_home",
            "FTAG": "goals_away",
            "FTR": "result",
            "HS": "hs",
            "AS": "as",
            "HST": "hst",
            "AST": "ast",
            "HF": "hf",
            "AF": "af",
            "HC": "hc",
            "AC": "ac",
            "HY": "hy",
            "AY": "ay",
            "HR": "hr",
            "AR": "ar",
        }
    )

    df["season"] = season
    df = df.sort_values("date").reset_index(drop=True)

    print(
        f"    {len(df)} matches, "
        f"{df['home_team'].nunique()} unique home teams"
    )
    return df


def build_clean_panel(save_to_csv: bool = True, force_rebuild: bool = False) -> pd.DataFrame:
    """
    Build (or reload) the full cleaned match panel for seasons 2020–2026.

    If `force_rebuild` is False and the cleaned file already exists, it is
    loaded from disk. Otherwise all raw season files are re-read and a new
    panel is built.

    Returns the cleaned match-level DataFrame.
    """
    if CLEAN_PANEL_FILE.exists() and not force_rebuild:
        print(f"Loading existing cleaned panel from {CLEAN_PANEL_FILE}")
        panel = pd.read_csv(CLEAN_PANEL_FILE, parse_dates=["date"])
        return panel

    print("Rebuilding cleaned match panel from raw CSV files...")
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    all_seasons: List[pd.DataFrame] = []
    for season, fname in SEASONS_TO_FILES.items():
        df_season = _load_one_season(season, fname)
        all_seasons.append(df_season)

    panel = pd.concat(all_seasons, ignore_index=True)
    panel = panel.sort_values("date").reset_index(drop=True)

    if save_to_csv:
        panel.to_csv(CLEAN_PANEL_FILE, index=False)
        print(f"✅ Saved cleaned panel to {CLEAN_PANEL_FILE}")

    print(
        f"Panel summary: {len(panel)} matches across "
        f"{sorted(panel['season'].unique())}"
    )
    return panel


# -------------------------------------------------------------------
# 2) Team–match feature engineering
# -------------------------------------------------------------------

def _expand_match_to_teams(row: pd.Series) -> Tuple[dict, dict]:
    """
    Convert a single match row into two rows: one for the home team and
    one for the away team.

    Returns two dictionaries suitable for constructing a DataFrame.
    """
    date = row["date"]
    season = row["season"]

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


def _compute_points(df: pd.DataFrame) -> pd.DataFrame:
    """Compute match points (3/1/0) for each team row."""
    df["pts"] = np.where(
        df["goals_for"] > df["goals_against"], 3,
        np.where(df["goals_for"] == df["goals_against"], 1, 0),
    )
    return df


def _compute_card_points(df: pd.DataFrame) -> pd.DataFrame:
    """Create a simple card penalty score (yellow = 1, red = 3)."""
    df["card_points"] = df["yellow"] * 1 + df["red"] * 3
    return df


def _compute_form(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Rolling average form over the last `window` matches."""
    df = df.sort_values(["team", "date"])
    df["form3"] = (
        df.groupby("team")["pts"]
        .rolling(window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return df


def _compute_rolling_opponent_strength(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """
    Approximate opponent strength: rolling average of opponent points
    over their last `window` matches (using realized pts).
    """
    opp_points = (
        df[["date", "team", "pts"]]
        .rename(columns={"team": "opponent", "pts": "opp_pts"})
    )

    merged = df.merge(opp_points, on=["opponent", "date"], how="left")
    merged["opp_avg_pts"] = (
        merged.groupby("team")["opp_pts"]
        .rolling(window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    merged = merged.drop(columns=["opp_pts"])
    return merged


def _compute_xg_diff(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple expected goals proxy using shots on target.
    This is a placeholder that can be replaced by real xG later.
    """
    df["xg_for"] = df["shots_on_target_for"] * 0.25
    df["xg_against"] = df["shots_on_target_against"] * 0.25
    df["xGD"] = df["xg_for"] - df["xg_against"]
    return df


def build_feature_dataset(
    save_to_csv: bool = True,
    force_rebuild_panel: bool = False,
) -> pd.DataFrame:
    """
    Build the team–match feature dataset used by the pricing and prediction
    parts of the project.

    Steps:
      1. Load or rebuild the cleaned match panel.
      2. Expand each match into two team rows (home and away).
      3. Compute points, card penalties, rolling form, opponent strength,
         and simple xG-based features.
      4. Save the resulting dataset to team_match_features.csv.

    Returns the team–match feature DataFrame.
    """
    panel = build_clean_panel(save_to_csv=True, force_rebuild=force_rebuild_panel)

    print("Creating team–match level feature dataset...")
    rows: List[dict] = []
    for _, row in panel.iterrows():
        home, away = _expand_match_to_teams(row)
        rows.append(home)
        rows.append(away)

    df = pd.DataFrame(rows)

    df = _compute_points(df)
    df = _compute_card_points(df)
    df = _compute_form(df)
    df = _compute_rolling_opponent_strength(df)
    df = _compute_xg_diff(df)

    df = df.sort_values(["team", "date"]).reset_index(drop=True)

    if save_to_csv:
        CLEAN_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(FEATURES_FILE, index=False)
        print(f"✅ Saved team–match features to {FEATURES_FILE}")

    print("Feature dataset sample:")
    print(df.head())
    return df


# -------------------------------------------------------------------
# 3) Per-season team summary (aggregated panel)
# -------------------------------------------------------------------

def build_season_summary(
    features: pd.DataFrame | None = None,
    save_to_csv: bool = True,
) -> pd.DataFrame:
    """
    Build a per-season team summary with aggregate statistics.

    If `features` is None, the function loads team_match_features.csv.

    Output columns (per season, team):
      matches, wins, draws, losses,
      goals_for, goals_against, goal_diff,
      points, points_per_game, xg_for, xg_against, xGD
    """
    if features is None:
        if not FEATURES_FILE.exists():
            raise FileNotFoundError(
                f"Feature file not found: {FEATURES_FILE}. "
                "Run build_feature_dataset() first."
            )
        features = pd.read_csv(FEATURES_FILE, parse_dates=["date"])

    print("Building per-season team summary...")

    # Basic aggregates
    grouped = (
        features.groupby(["season", "team"])
        .agg(
            matches=("pts", "size"),
            points=("pts", "sum"),
            goals_for=("goals_for", "sum"),
            goals_against=("goals_against", "sum"),
            xg_for=("xg_for", "sum"),
            xg_against=("xg_against", "sum"),
        )
        .reset_index()
    )

    grouped["goal_diff"] = grouped["goals_for"] - grouped["goals_against"]
    grouped["xGD"] = grouped["xg_for"] - grouped["xg_against"]
    grouped["points_per_game"] = grouped["points"] / grouped["matches"]

    # Wins / draws / losses
    wdl = (
        features.assign(
            win=features["pts"].eq(3).astype(int),
            draw=features["pts"].eq(1).astype(int),
            loss=features["pts"].eq(0).astype(int),
        )
        .groupby(["season", "team"])
        .agg(wins=("win", "sum"), draws=("draw", "sum"), losses=("loss", "sum"))
        .reset_index()
    )

    summary = grouped.merge(wdl, on=["season", "team"], how="left")

    if save_to_csv:
        summary.to_csv(SEASON_SUMMARY_FILE, index=False)
        print(f"✅ Saved season summary to {SEASON_SUMMARY_FILE}")

    print("Season summary sample:")
    print(summary.head())
    return summary


# -------------------------------------------------------------------
# 4) Command-line entry point
# -------------------------------------------------------------------

@dataclass
class DataLoaderOutputs:
    panel: pd.DataFrame
    features: pd.DataFrame
    season_summary: pd.DataFrame


def run_full_pipeline(
    save_to_csv: bool = True,
    force_rebuild_panel: bool = False,
) -> DataLoaderOutputs:
    """
    Convenience function to run the entire data pipeline:

      - cleaned match panel
      - team–match feature dataset
      - per-season team summary
    """
    print("=== Running data_loader pipeline ===")

    panel = build_clean_panel(save_to_csv=save_to_csv, force_rebuild=force_rebuild_panel)
    features = build_feature_dataset(save_to_csv=save_to_csv, force_rebuild_panel=False)
    season_summary = build_season_summary(features=features, save_to_csv=save_to_csv)

    print("=== Data pipeline complete ===")
    return DataLoaderOutputs(
        panel=panel,
        features=features,
        season_summary=season_summary,
    )


if __name__ == "__main__":
    # When you run: `python3 src/data_loader.py`  (from project root)
    # or          : `python3 -m data_loader`     (from inside src/)
    # this will execute the full pipeline.
    run_full_pipeline(save_to_csv=True, force_rebuild_panel=False)