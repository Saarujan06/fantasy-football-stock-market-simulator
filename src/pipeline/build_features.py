from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# =============================================================================
# PATHS
# =============================================================================

# File: src/pipeline/build_features.py
# parents[0]=pipeline, [1]=src, [2]=project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned_data"
CLEANED_DIR.mkdir(parents=True, exist_ok=True)

UNDERSTAT_PATH = RAW_DATA_DIR / "understat_data" / "game_stats.csv"
OUTPUT_MATCH_LEVEL = CLEANED_DIR / "team_match_features.csv"

# =============================================================================
# TEAM NAME NORMALISATION (SHARED)
# =============================================================================

from src.utils.team_names import normalize_team_name

# =============================================================================
# HELPERS
# =============================================================================

def season_str_from_start_year(start_year: int) -> str:
    """Map Understat numeric season (e.g. 2014) → '2014_2015'."""
    return f"{start_year}_{start_year + 1}"

# =============================================================================
# 1. LOAD FOOTBALL-DATA (RESULTS + ODDS)
# =============================================================================

def load_football_data(fd_dir: Path) -> pd.DataFrame:
    csv_files = sorted(fd_dir.glob("prem_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No prem_*.csv files found in {fd_dir}")

    print("Loading & parsing Football-Data season files ...")
    season_rows: List[pd.DataFrame] = []

    for f in csv_files:
        print(f"  - {f.name}")
        parts = f.stem.split("_")  # prem_2010_2011
        season = f"{parts[1]}_{parts[2]}" if len(parts) >= 3 else None

        df = pd.read_csv(f)

        required_cols = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}
        missing = required_cols - set(df.columns)
        if missing:
            raise KeyError(f"{f.name} missing columns: {missing}")

        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["Date"])

        df["Season"] = season if season else df["Season"].astype(str)
        season_rows.append(df)

    all_matches = pd.concat(season_rows, ignore_index=True)
    all_matches = all_matches.sort_values(["Season", "Date"]).reset_index(drop=True)

    print(f"Total match rows (Football-Data): {len(all_matches)}")
    return all_matches

# =============================================================================
# 2. TEAM-LEVEL PANEL
# =============================================================================

def to_team_level(matches: pd.DataFrame) -> pd.DataFrame:
    m = matches.copy()

    m["HomeTeam"] = m["HomeTeam"].apply(normalize_team_name)
    m["AwayTeam"] = m["AwayTeam"].apply(normalize_team_name)

    home = pd.DataFrame({
        "season": m["Season"],
        "date": m["Date"].dt.normalize(),
        "team": m["HomeTeam"],
        "opponent": m["AwayTeam"],
        "home_away": "H",
        "goals_for": m["FTHG"],
        "goals_against": m["FTAG"],
        "result": m["FTR"].map({"H": 2, "D": 1, "A": 0}),
    })

    away = pd.DataFrame({
        "season": m["Season"],
        "date": m["Date"].dt.normalize(),
        "team": m["AwayTeam"],
        "opponent": m["HomeTeam"],
        "home_away": "A",
        "goals_for": m["FTAG"],
        "goals_against": m["FTHG"],
        "result": m["FTR"].map({"H": 0, "D": 1, "A": 2}),
    })

    team_panel = pd.concat([home, away], ignore_index=True)

    for col in ["B365H", "B365D", "B365A"]:
        team_panel[col] = pd.concat(
            [m[col], m[col]], ignore_index=True
        ) if col in m.columns else np.nan

    team_panel = team_panel.sort_values(
        ["season", "date", "team"]
    ).reset_index(drop=True)

    print(f"Team-level match rows: {len(team_panel)}")
    return team_panel

# =============================================================================
# 3. BETTING FEATURES
# =============================================================================

def _safe_inverse_odds(odds: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(odds, errors="coerce").to_numpy()
    return np.where(arr > 0, 1.0 / arr, np.nan)

def add_betting_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    h = _safe_inverse_odds(out["B365H"])
    d = _safe_inverse_odds(out["B365D"])
    a = _safe_inverse_odds(out["B365A"])
    total = h + d + a

    out["b365_home_prob"] = h / total
    out["b365_draw_prob"] = d / total
    out["b365_away_prob"] = a / total
    out["b365_team_win_prob"] = np.where(
        out["home_away"] == "H",
        out["b365_home_prob"],
        out["b365_away_prob"],
    )

    return out

# =============================================================================
# 4. LOAD UNDERSTAT
# =============================================================================

def load_understat(path: Path) -> pd.DataFrame:
    us = pd.read_csv(path)

    us = us[us["league"].str.contains("premier|epl", case=False, na=False)]
    us["date"] = pd.to_datetime(us["date"], errors="coerce").dt.normalize()
    us["season"] = us["season"].astype(int).apply(season_str_from_start_year)
    us["team"] = us["club_name"].apply(normalize_team_name)
    us["home_away"] = us["home_away"].str.upper().str[0]

    keep = [
        "season", "date", "team", "home_away",
        "xG", "xGA", "npxG", "npxGA",
        "ppda", "ppda_allowed", "deep", "deep_allowed",
        "scored", "missed", "xpts",
        "wins", "draws", "loses", "pts", "npxGD",
    ]

    us = us[keep].sort_values(["season", "team", "date"])
    print(f"Understat team rows: {len(us)}")
    return us

# =============================================================================
# 5. MERGE UNDERSTAT
# =============================================================================

def merge_understat(team_panel: pd.DataFrame, us: pd.DataFrame) -> pd.DataFrame:
    df = team_panel.copy()
    df["date"] = pd.to_datetime(df["date"])
    us["date"] = pd.to_datetime(us["date"])

    merged = pd.merge(
        df,
        us,
        how="left",
        on=["season", "date", "team", "home_away"],
    )

    print(f"Rows with xG: {merged['xG'].notna().sum()} / {len(merged)}")
    return merged

# =============================================================================
# 6. FEATURE ENGINEERING (LEAKAGE SAFE)
# =============================================================================

def _rolling_past(df, value, name, window):
    return (
        df.groupby("team")[value]
        .shift(1)
        .rolling(window, min_periods=1)
        .mean()
        .rename(name)
    )

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["label"] = df["result"].astype(int)
    df["points"] = df["label"].map({0: 0, 1: 1, 2: 3})

    for w in [3, 5]:
        df[f"xG_last{w}"] = _rolling_past(df, "xG", f"xG_last{w}", w)
        df[f"xGA_last{w}"] = _rolling_past(df, "xGA", f"xGA_last{w}", w)
        df[f"pts_last{w}"] = _rolling_past(df, "points", f"pts_last{w}", w)

    drop_cols = [
        "goals_for", "goals_against", "points",
        "xG", "xGA", "npxG", "npxGA",
        "ppda", "ppda_allowed", "deep", "deep_allowed",
        "scored", "missed", "xpts",
        "wins", "draws", "loses", "pts", "npxGD",
        "result",
    ]

    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df = df.fillna(0.0)

    print(f"Final rows after feature engineering: {len(df)}")
    return df

# =============================================================================
# 7. BUILD PIPELINE
# =============================================================================

def build() -> pd.DataFrame:
    matches = load_football_data(RAW_DATA_DIR)
    panel = to_team_level(matches)
    panel = add_betting_features(panel)

    print("Loading Understat...")
    us = load_understat(UNDERSTAT_PATH)

    print("Merging datasets...")
    merged = merge_understat(panel, us)

    print("Engineering features...")
    features = engineer_features(merged)

    features.to_csv(OUTPUT_MATCH_LEVEL, index=False)
    print(f"Saved: {OUTPUT_MATCH_LEVEL}")

    return features

if __name__ == "__main__":
    build()