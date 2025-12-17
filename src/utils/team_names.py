# src/utils/team_names.py
from __future__ import annotations

from typing import Dict, Iterable, Set
import pandas as pd

# =============================================================================
# Canonical teams for the CURRENT Premier League season (2025–26)
# =============================================================================

CANONICAL_TEAMS_2526: Set[str] = {
    "Arsenal",
    "Aston Villa",
    "Bournemouth",
    "Brentford",
    "Brighton and Hove Albion",
    "Burnley",
    "Chelsea",
    "Crystal Palace",
    "Everton",
    "Fulham",
    "Leeds United",
    "Liverpool",
    "Manchester City",
    "Manchester United",
    "Newcastle United",
    "Nottingham Forest",
    "Sunderland",
    "Tottenham Hotspur",
    "West Ham United",
    "Wolverhampton Wanderers",
}

# =============================================================================
# Name normalization map 
# =============================================================================

TEAM_NAME_MAP: Dict[str, str] = {
    # Manchester United
    "Man United": "Manchester United",
    "Man Utd": "Manchester United",
    "Manchester Utd": "Manchester United",

    # Manchester City
    "Man City": "Manchester City",
    "Man. City": "Manchester City",

    # Tottenham
    "Spurs": "Tottenham Hotspur",
    "Tottenham": "Tottenham Hotspur",
    "Tottenham Hotspur": "Tottenham Hotspur",

    # Wolves
    "Wolves": "Wolverhampton Wanderers",
    "Wolverhampton": "Wolverhampton Wanderers",
    "Wolverhampton Wanderers": "Wolverhampton Wanderers",

    # West Ham
    "West Ham": "West Ham United",
    "West Ham Utd": "West Ham United",
    "West Ham United": "West Ham United",

    # Newcastle
    "Newcastle": "Newcastle United",
    "Newcastle Utd": "Newcastle United",
    "Newcastle United": "Newcastle United",

    # Brighton
    "Brighton": "Brighton and Hove Albion",
    "Brighton & Hove Albion": "Brighton and Hove Albion",
    "Brighton and Hove Albion": "Brighton and Hove Albion",

    # Forest
    "Nott'm Forest": "Nottingham Forest",
    "Nottingham Forest": "Nottingham Forest",

    # Leeds
    "Leeds": "Leeds United",
    "Leeds United": "Leeds United",

    # Bournemouth
    "AFC Bournemouth": "Bournemouth",
    "Bournemouth": "Bournemouth",

    # --- Historical / promoted / relegated teams (KEEP THESE) ---
    "West Brom": "West Bromwich Albion",
    "West Bromwich": "West Bromwich Albion",
    "West Bromwich Albion": "West Bromwich Albion",

    "Stoke": "Stoke City",
    "Stoke City": "Stoke City",

    "Hull": "Hull City",
    "Hull City": "Hull City",

    "Cardiff": "Cardiff City",
    "Cardiff City": "Cardiff City",

    "Huddersfield": "Huddersfield Town",
    "Huddersfield Town": "Huddersfield Town",

    "Norwich": "Norwich City",
    "Norwich City": "Norwich City",

    "Swansea": "Swansea City",
    "Swansea City": "Swansea City",

    "QPR": "Queens Park Rangers",
    "Queens Park Rangers": "Queens Park Rangers",

    "Leicester": "Leicester City",
    "Leicester City": "Leicester City",

    # If you see new ones in your CSVs, just add them here.
}

# =============================================================================
# Public helpers
# =============================================================================

def normalize_team_name(name: object) -> object:
    """
    Normalize a team name to your canonical naming scheme.
    - Safe on NaNs (returns as-is)
    - Safe on non-strings (casts to str)
    """
    if pd.isna(name):
        return name
    s = str(name).strip()
    return TEAM_NAME_MAP.get(s, s)


def normalize_team_series(s: pd.Series) -> pd.Series:
    """Vectorized normalization for a pandas Series."""
    return s.map(normalize_team_name)


def is_current_pl_team_2526(name: object) -> bool:
    """True iff name normalizes into the 2025–26 PL canonical set."""
    if pd.isna(name):
        return False
    return normalize_team_name(name) in CANONICAL_TEAMS_2526


def filter_current_pl_2526(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """
    Filter a DataFrame to ONLY rows where ALL provided team columns are
    in CANONICAL_TEAMS_2526 after normalization.

    Example:
        df = filter_current_pl_2526(df, ["home_team", "away_team"])
    """
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            raise KeyError(f"Column '{c}' not in DataFrame: {list(out.columns)}")
        out[c] = out[c].apply(normalize_team_name)

    mask = pd.Series(True, index=out.index)
    for c in cols:
        mask &= out[c].isin(CANONICAL_TEAMS_2526)

    return out[mask].copy()