from __future__ import annotations
from pathlib import Path
from typing import List, Dict
import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "data" / "cleaned_data"

OUT_FILE = OUT_DIR / "Full_Prem_2020-2026_cleaned.csv"


# ---------------------------------------------------------
# Helper: load one season
# ---------------------------------------------------------
def load_one_season(season: str, filename: str) -> pd.DataFrame:
    """
    Load a single Premier League season from a football-data.co.uk style CSV.

    Expected columns in the raw file:
      - Date
      - HomeTeam
      - AwayTeam
      - FTHG, FTAG, FTR
      - HS, AS, HST, AST, HF, AF, HC, AC, HY, AY, HR, AR

    Returns a DataFrame with unified, snake_case columns + a 'season' column.
    """
    path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Raw file not found for season {season}: {path}")

    print(f"Loading {season} from {path} ...")
    df = pd.read_csv(path, encoding="latin1")

    # Some files may have extra columns; we only keep what we need
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
            f"File {filename} for season {season} is missing columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    df = df[required_cols].copy()

    # Parse dates (football-data uses day-first format like '12/09/2020')
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    before = len(df)
    df = df.dropna(subset=["Date"])
    after = len(df)
    if after < before:
        print(f"  Dropped {before - after} rows with invalid dates in {season}")

    # Rename to snake_case
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

    # Add season column
    df["season"] = season

    # Optional: sort within season by date
    df = df.sort_values("date").reset_index(drop=True)

    print(
        f"  Loaded {len(df)} matches for {season} "
        f"({df['home_team'].nunique()} home teams, "
        f"{df['away_team'].nunique()} away teams)"
    )
    return df


# ---------------------------------------------------------
# Main builder
# ---------------------------------------------------------
def build_clean_panel() -> pd.DataFrame:
    """
    Build a single cleaned match-level dataset for all seasons 2020–2026.

    Uses raw CSVs in data/raw and saves:
      data/cleaned_data/Full_Prem_2020-2026_cleaned.csv
    """
    seasons_to_files: Dict[str, str] = {
        "2020-2021": "prem_2020_2021.csv",
        "2021-2022": "prem_2021_2022.csv",
        "2022-2023": "prem_2022_2023.csv",
        "2023-2024": "prem_2023_2024.csv",
        "2024-2025": "prem_2024_2025.csv",
        "2025-2026": "prem_2025_2026.csv",
    }

    all_seasons: List[pd.DataFrame] = []

    for season, fname in seasons_to_files.items():
        df_season = load_one_season(season, fname)
        all_seasons.append(df_season)

    full = pd.concat(all_seasons, ignore_index=True)

    # Global sort by date just to have a clean timeline
    full = full.sort_values("date").reset_index(drop=True)

    # Ensure output directory exists
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    full.to_csv(OUT_FILE, index=False)

    print("\n======================================")
    print("✅ Built cleaned full panel:")
    print(f"  Saved to: {OUT_FILE}")
    print(f"  Total rows: {len(full)}")
    print(f"  Seasons: {sorted(full['season'].unique())}")
    print(f"  Teams (home): {full['home_team'].nunique()}")
    print(f"  Teams (away): {full['away_team'].nunique()}")
    print("======================================\n")

    return full


if __name__ == "__main__":
    build_clean_panel()