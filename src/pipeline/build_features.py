from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# =============================================================================
# PATHS
# =============================================================================

# This file lives in src/pipeline/build_features.py
# parents[0] = pipeline, [1] = src, [2] = project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned_data"
CLEANED_DIR.mkdir(parents=True, exist_ok=True)

UNDERSTAT_PATH = RAW_DATA_DIR / "understat-data" / "game_stats.csv"
OUTPUT_MATCH_LEVEL = CLEANED_DIR / "team_match_features.csv"

# =============================================================================
# UTILS
# =============================================================================

TEAM_NAME_MAP: Dict[str, str] = {
    "Man United": "Manchester United",
    "Man Utd": "Manchester United",
    "Manchester Utd": "Manchester United",
    "Man City": "Manchester City",
    "Man. City": "Manchester City",
    "Spurs": "Tottenham",
    "Tottenham Hotspur": "Tottenham",
    "Wolves": "Wolverhampton",
    "Wolverhampton Wanderers": "Wolverhampton",
    "West Bromwich Albion": "West Brom",
    "Brighton and Hove Albion": "Brighton",
    "Brighton & Hove Albion": "Brighton",
    "Newcastle Utd": "Newcastle",
    "Newcastle United": "Newcastle",
    "Nott'm Forest": "Nottingham Forest",
    "Nottingham Forest": "Nottingham Forest",
    "Huddersfield Town": "Huddersfield",
    "Cardiff City": "Cardiff",
    "Norwich City": "Norwich",
    "Swansea City": "Swansea",
    "Leicester City": "Leicester",
    "Stoke City": "Stoke",
    "Hull City": "Hull",
    "QPR": "Queens Park Rangers",
    "Queens Park Rangers": "Queens Park Rangers",
    "AFC Bournemouth": "Bournemouth",
    "Sheffield Utd": "Sheffield United",
    "Sheffield United": "Sheffield United",
    "West Ham United": "West Ham",
    "West Ham": "West Ham",
}


def normalize_team_name(name: str) -> str:
    if pd.isna(name):
        return name
    return TEAM_NAME_MAP.get(name, name)


def season_str_from_start_year(start_year: int) -> str:
    """Map Understat numeric season (e.g. 2014) → '2014_2015'."""
    return f"{start_year}_{start_year + 1}"


# =============================================================================
# 1. LOAD FOOTBALL-DATA (RESULTS + ODDS)
# =============================================================================

def load_football_data(fd_dir: Path) -> pd.DataFrame:
    """Load all prem_YYYY_YYYY.csv files and return one match-level DataFrame."""
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
            raise KeyError(f"{f.name} is missing required columns: {missing}")

        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["Date"])

        if season is not None:
            df["Season"] = season
        else:
            if "Season" not in df.columns:
                raise KeyError(
                    f"Could not infer season for {f.name} and no 'Season' column present."
                )
            df["Season"] = df["Season"].astype(str)

        season_rows.append(df)

    all_matches = pd.concat(season_rows, ignore_index=True)
    all_matches = all_matches.sort_values(["Season", "Date"]).reset_index(drop=True)

    print(f"Total match rows (Football-Data): {len(all_matches)}")
    return all_matches


# =============================================================================
# 2. CONVERT TO TEAM-LEVEL PANEL
# =============================================================================

def to_team_level(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Create team-level panel: one row per team per match.

    Columns:
      season, date, team, opponent, home_away, goals_for, goals_against, result
      plus odds (B365H/B365D/B365A) if present.
    """
    m = matches.copy()
    m["HomeTeam"] = m["HomeTeam"].apply(normalize_team_name)
    m["AwayTeam"] = m["AwayTeam"].apply(normalize_team_name)

    home_rows = pd.DataFrame(
        {
            "season": m["Season"].astype(str),
            "date": m["Date"].dt.normalize(),
            "team": m["HomeTeam"],
            "opponent": m["AwayTeam"],
            "home_away": "H",
            "goals_for": m["FTHG"],
            "goals_against": m["FTAG"],
            "result": m["FTR"].map({"H": 2, "D": 1, "A": 0}),
        }
    )

    away_rows = pd.DataFrame(
        {
            "season": m["Season"].astype(str),
            "date": m["Date"].dt.normalize(),
            "team": m["AwayTeam"],
            "opponent": m["HomeTeam"],
            "home_away": "A",
            "goals_for": m["FTAG"],
            "goals_against": m["FTHG"],
            "result": m["FTR"].map({"H": 0, "D": 1, "A": 2}),
        }
    )

    team_panel = pd.concat([home_rows, away_rows], ignore_index=True)

    for col in ["B365H", "B365D", "B365A"]:
        if col in m.columns:
            team_panel[col] = pd.concat([m[col], m[col]], ignore_index=True)
        else:
            team_panel[col] = np.nan

    team_panel = team_panel.sort_values(["season", "date", "team"]).reset_index(drop=True)
    print(f"Team-level match rows: {len(team_panel)}")
    return team_panel


# =============================================================================
# 3. BETTING FEATURES
# =============================================================================

def _safe_inverse_odds(odds: pd.Series) -> np.ndarray:
    odds_arr = pd.to_numeric(odds, errors="coerce").to_numpy()
    return np.where(odds_arr > 0, 1.0 / odds_arr, np.nan)


def add_betting_features(team_panel: pd.DataFrame) -> pd.DataFrame:
    """From B365 odds compute implied probs and team win prob."""
    df = team_panel.copy()

    h = _safe_inverse_odds(df["B365H"])
    d = _safe_inverse_odds(df["B365D"])
    a = _safe_inverse_odds(df["B365A"])
    total = h + d + a

    df["b365_home_prob"] = h / total
    df["b365_draw_prob"] = d / total
    df["b365_away_prob"] = a / total
    df["b365_team_win_prob"] = np.where(df["home_away"] == "H", df["b365_home_prob"], df["b365_away_prob"])

    return df


# =============================================================================
# 4. LOAD UNDERSTAT & MERGE (ROBUST, NO EXACT-DATE REQUIREMENT)
# =============================================================================

def load_understat(path: Path) -> pd.DataFrame:
    """Load Understat team-level game_stats for EPL-like leagues."""
    fp = Path(path)
    if not fp.exists():
        raise FileNotFoundError(f"Understat game_stats CSV not found at: {fp}")

    us = pd.read_csv(fp)

    if "league" in us.columns:
        mask = us["league"].astype(str).str.contains("premier|epl", case=False, regex=True)
        us = us[mask].copy()

    if "date" not in us.columns:
        raise KeyError("Understat data must contain a 'date' column.")
    us["date"] = pd.to_datetime(us["date"], errors="coerce")
    us = us.dropna(subset=["date"])
    us["date"] = us["date"].dt.normalize()

    if "season" not in us.columns:
        raise KeyError("Understat data must contain 'season' column.")
    us["season"] = us["season"].astype(int)
    us["season"] = us["season"].apply(season_str_from_start_year).astype(str)

    if "club_name" not in us.columns:
        raise KeyError("Understat data must contain 'club_name' column.")
    us["team"] = us["club_name"].apply(normalize_team_name)

    if "home_away" not in us.columns:
        raise KeyError("Understat data must contain 'home_away' column.")
    us["home_away"] = us["home_away"].astype(str).str.strip().str[0].str.upper()

    keep_cols = [
        "season",
        "date",
        "team",
        "home_away",
        "xG",
        "xGA",
        "npxG",
        "npxGA",
        "ppda",
        "ppda_allowed",
        "deep",
        "deep_allowed",
        "scored",
        "missed",
        "xpts",
        "wins",
        "draws",
        "loses",
        "pts",
        "npxGD",
    ]
    missing = [c for c in keep_cols if c not in us.columns]
    if missing:
        raise KeyError(f"Missing expected Understat columns: {missing}")

    us_team = us[keep_cols].copy()
    us_team = us_team.sort_values(["season", "team", "home_away", "date"]).reset_index(drop=True)

    print(f"Understat team rows (EPL-like): {len(us_team)}")
    return us_team


def merge_understat(team_panel: pd.DataFrame, us_team: pd.DataFrame) -> pd.DataFrame:
    df = team_panel.copy()

    # Normalize join keys
    df["season"] = df["season"].astype(str)
    df["team"] = df["team"].astype(str)
    df["home_away"] = df["home_away"].astype(str).str.strip().str.upper()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["season", "team", "home_away", "date"])

    us = us_team.copy()
    us["season"] = us["season"].astype(str)
    us["team"] = us["team"].astype(str)
    us["home_away"] = us["home_away"].astype(str).str.strip().str.upper()
    us["date"] = pd.to_datetime(us["date"], errors="coerce").dt.normalize()
    us = us.dropna(subset=["season", "team", "home_away", "date"])

    # Optional: restrict Understat to seasons we have
    us = us[us["season"].isin(df["season"].unique())].copy()

    merged = pd.merge(
        df,
        us,
        how="left",
        on=["season", "date", "team", "home_away"],
        suffixes=("", "_us"),
    )

    n_with_xg = merged["xG"].notna().sum() if "xG" in merged.columns else 0
    total = len(merged)
    pct = (n_with_xg / total * 100) if total else 0.0
    print(f"Understat rows merged onto team panel: {n_with_xg} ({pct:.1f}% of rows have xG)")

    return merged


# =============================================================================
# 5. ENGINEER FEATURES (NO LEAKAGE) + ROLLING xG
# =============================================================================

def _rolling_past_by_team(
    df: pd.DataFrame,
    group_cols: List[str],
    sort_cols: List[str],
    value_col: str,
    new_col: str,
    window: int,
) -> pd.DataFrame:
    """Rolling mean using only past values (shift(1))."""
    df = df.sort_values(group_cols + sort_cols).copy()
    grouped = df.groupby(group_cols, group_keys=False)

    df[new_col] = grouped[value_col].apply(
        lambda s: s.shift(1).rolling(window=window, min_periods=1).mean()
    )
    return df


def engineer_features(team_with_us: pd.DataFrame) -> pd.DataFrame:
    """
    Create leakage-safe rolling form features.
    IMPORTANT: We compute rolling xG from *past matches only* (shifted).
    """
    df = team_with_us.copy()

    df = df.dropna(subset=["result"])
    df["result"] = df["result"].astype(int)
    df["label"] = df["result"]

    df["points"] = df["result"].map({0: 0, 1: 1, 2: 3})

    windows = [3, 5]
    for w in windows:
        df = _rolling_past_by_team(df, ["team"], ["season", "date"], "goals_for", f"goals_for_last{w}", w)
        df = _rolling_past_by_team(df, ["team"], ["season", "date"], "goals_against", f"goals_against_last{w}", w)
        df = _rolling_past_by_team(df, ["team"], ["season", "date"], "points", f"points_last{w}", w)

        # Rolling xG (past-only) if available from Understat merge
        if "xG" in df.columns:
            df = _rolling_past_by_team(df, ["team"], ["season", "date"], "xG", f"xG_last{w}", w)
        if "xGA" in df.columns:
            df = _rolling_past_by_team(df, ["team"], ["season", "date"], "xGA", f"xGA_last{w}", w)

        if "b365_team_win_prob" in df.columns:
            df = _rolling_past_by_team(
                df, ["team"], ["season", "date"], "b365_team_win_prob", f"b365_team_win_prob_last{w}", w
            )

    # Friendly alias names (still leakage-safe)
    if "xG_last3" in df.columns:
        df["xG_for_last3"] = df["xG_last3"]
    if "xG_last5" in df.columns:
        df["xG_for_last5"] = df["xG_last5"]
    if "xGA_last3" in df.columns:
        df["xG_against_last3"] = df["xGA_last3"]
    if "xGA_last5" in df.columns:
        df["xG_against_last5"] = df["xGA_last5"]

    df = df.dropna(subset=["label", "date"])
    df = df.sort_values(["season", "date", "team"]).reset_index(drop=True)

    # Drop leakage columns (current-match realized outcomes / same-match xG)
    leak_cols = [
        "goals_for",
        "goals_against",
        "points",
        "xG",
        "xGA",
        "npxG",
        "npxGA",
        "ppda",
        "ppda_allowed",
        "deep",
        "deep_allowed",
        "scored",
        "missed",
        "xpts",
        "wins",
        "draws",
        "loses",
        "pts",
        "npxGD",
        "result",
    ]
    leak_cols_present = [c for c in leak_cols if c in df.columns]
    df = df.drop(columns=leak_cols_present)

    # Fill numeric NaNs (early season, promoted teams, pre-2014 missing xG, etc.)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[num_cols] = df[num_cols].fillna(0.0)

    print(f"Filled NaNs in numeric features with 0. Remaining rows: {len(df)}")
    return df


# =============================================================================
# 6. MAIN BUILD FUNCTION
# =============================================================================

def build() -> pd.DataFrame:
    """Run full pipeline and save team_match_features.csv."""
    matches = load_football_data(RAW_DATA_DIR)
    team_panel = to_team_level(matches)
    team_panel = add_betting_features(team_panel)

    print("Loading Understat game_stats ...")
    understat_team = load_understat(UNDERSTAT_PATH)

    print("Merging Football-Data panel with Understat team metrics ...")
    merged = merge_understat(team_panel, understat_team)

    print("Adding engineered features (form, lags, betting, rolling xG, etc.) ...")
    match_features = engineer_features(merged)

    OUTPUT_MATCH_LEVEL.parent.mkdir(parents=True, exist_ok=True)
    match_features.to_csv(OUTPUT_MATCH_LEVEL, index=False)
    print(f"Saved match-level features to: {OUTPUT_MATCH_LEVEL}")

    print(
        f"Final dataset: {len(match_features)} rows, "
        f"{match_features.shape[1]} columns, "
        f"NaNs total: {match_features.isna().sum().sum()}"
    )

    return match_features


if __name__ == "__main__":
    build()