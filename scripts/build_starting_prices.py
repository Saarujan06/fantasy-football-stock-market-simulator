from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEAN_DIR = PROJECT_ROOT / "data" / "cleaned_data"

START_PRICES_FILE = CLEAN_DIR / "starting_prices_2019.csv"


# ---------------------------------------------------------
# 1. Default historical baseline table
# ---------------------------------------------------------
def _default_starting_prices_df() -> pd.DataFrame:
    """
    Realistic baseline derived from the 2018/19 Premier League table.

    These values act as the "IPO" prices for all teams.
    Promoted teams will get adjusted IPO prices later.
    """
    data = [
        ("Manchester City", 128),
        ("Liverpool", 126),
        ("Chelsea", 123),
        ("Tottenham Hotspur", 120),
        ("Arsenal", 118),
        ("Manchester United", 116),
        ("Wolverhampton Wanderers", 113),
        ("Everton", 110),
        ("West Ham United", 105),
        ("Crystal Palace", 100),
        ("Newcastle United", 98),
        ("Bournemouth", 95),
        ("Burnley", 93),
        ("Brighton and Hove Albion", 88),
        ("Aston Villa", 83),
        ("Fulham", 84),
        ("Leeds United", 82),
        ("Nottingham Forest", 81),
        ("Brentford", 80),
        ("Sunderland", 78),
    ]
    return pd.DataFrame(data, columns=["team", "start_price"])


# ---------------------------------------------------------
# 2. Write the default baseline to disk
# ---------------------------------------------------------
def _write_default_start_prices() -> None:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    df_default = _default_starting_prices_df()
    df_default.to_csv(START_PRICES_FILE, index=False)
    print(f"✅ Created default starting prices → {START_PRICES_FILE}")


# ---------------------------------------------------------
# 3. Load starting prices with auto-repair
# ---------------------------------------------------------
def load_start_prices(teams: pd.Index) -> Dict[str, float]:
    """
    Load or automatically rebuild starting_prices_2019.csv.

    Behaviours:
      - If the file does not exist → create it.
      - If the file exists but has wrong columns → overwrite with default.
      - If some teams are missing (e.g., newly promoted teams) →
        assign IPO-style starting prices below the lowest historical team.
    """

    # If missing → create a new good file
    if not START_PRICES_FILE.exists():
        _write_default_start_prices()

    # Load file (it may be wrong)
    sp = pd.read_csv(START_PRICES_FILE)

    # Wrong schema → overwrite file
    if not {"team", "start_price"}.issubset(sp.columns):
        print(
            f"⚠️ Invalid starting price file detected ({sp.columns.tolist()}). "
            f"Rebuilding file..."
        )
        _write_default_start_prices()
        sp = pd.read_csv(START_PRICES_FILE)

    # Keep only teams in current dataset
    sp = sp[sp["team"].isin(teams)]

    start_map: Dict[str, float] = {}

    # Add all known teams
    for _, row in sp.iterrows():
        start_map[row["team"]] = float(row["start_price"])

    # ---------------------------------------------------------
    # Handle promoted teams: IPO logic
    # ---------------------------------------------------------
    missing_teams = [t for t in teams if t not in start_map]

    if missing_teams:
        lowest_price = min(start_map.values()) if start_map else 78  # fallback

        print(f"🏆 Assigning IPO-style prices to promoted teams: {missing_teams}")

        for i, t in enumerate(missing_teams, start=1):
            start_map[t] = max(lowest_price - 2 * i, 50.0)  # avoid tiny IPOs

    return start_map