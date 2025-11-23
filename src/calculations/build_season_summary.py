from __future__ import annotations
from pathlib import Path
import pandas as pd


# --------------------------------------------------------------
# Paths
# --------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "cleaned_data"

MATCH_PANEL_FILE = DATA_DIR / "team_match_features.csv"
OUT_FILE = DATA_DIR / "season_summary.csv"


# --------------------------------------------------------------
# Load team-match panel
# --------------------------------------------------------------
def load_match_panel() -> pd.DataFrame:
    if not MATCH_PANEL_FILE.exists():
        raise FileNotFoundError(
            f"Missing match panel: {MATCH_PANEL_FILE}\n"
            "Run feature_engineering.py first."
        )

    df = pd.read_csv(MATCH_PANEL_FILE)
    df.columns = [c.strip() for c in df.columns]

    # Ensure required columns exist
    needed = {"team", "season", "goals_for", "goals_against", "pts", "date"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in match panel: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"])
    return df


# --------------------------------------------------------------
# Build per-season summary
# --------------------------------------------------------------
def build_season_summary() -> pd.DataFrame:
    df = load_match_panel()

    print(f"Loaded {len(df)} match rows for {df['team'].nunique()} teams.")

    # 1) Aggregate points + goal difference per season
    grouped = df.groupby(["team", "season"]).agg(
        points=("pts", "sum"),
        goals_for=("goals_for", "sum"),
        goals_against=("goals_against", "sum"),
    ).reset_index()

    grouped["gd"] = grouped["goals_for"] - grouped["goals_against"]

    # 2) Compute league positions PER SEASON
    def assign_positions(g: pd.DataFrame) -> pd.DataFrame:
        # Higher points = better rank
        g = g.sort_values(
            ["points", "gd", "goals_for"],
            ascending=[False, False, False]
        ).reset_index(drop=True)
        g["position"] = range(1, len(g) + 1)
        return g

    ranked = grouped.groupby("season", group_keys=False).apply(assign_positions)

    # 3) Optional: sort nicely
    ranked = ranked.sort_values(["season", "position"])

    # 4) Save
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(OUT_FILE, index=False)

    print(f"✅ Saved season summary to: {OUT_FILE}")
    print("\n===== SAMPLE SEASON SUMMARY =====")
    print(ranked.head(20))

    return ranked


# --------------------------------------------------------------
# CLI
# --------------------------------------------------------------
if __name__ == "__main__":
    build_season_summary()