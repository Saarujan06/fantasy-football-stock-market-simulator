from __future__ import annotations
from pathlib import Path
import pandas as pd

# -----------------------------------------------------
# Paths
# -----------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEANED_DIR = PROJECT_ROOT / "data" / "Cleaned Data"
FEATURES_OUT = CLEANED_DIR / "team_match_features.csv"

# -----------------------------------------------------
# File lists
# -----------------------------------------------------
PREM_FILES = [
    CLEANED_DIR / "Prem 2020:2021 (cleaned).csv",
    CLEANED_DIR / "Prem 2021:2022 (cleaned).csv",
    CLEANED_DIR / "Prem 2022:2023 (cleaned).csv",
    CLEANED_DIR / "Prem 2023:2024 (cleaned).csv",
    CLEANED_DIR / "Prem 2024:2025 (cleaned).csv",
    CLEANED_DIR / "Prem 2025:2026 (cleaned).csv",
]

UNDERSTAT_FILES = [
    CLEANED_DIR / "cleaned_understat_2020.csv",
    CLEANED_DIR / "cleaned_understat_2021.csv",
    CLEANED_DIR / "cleaned_understat_2022.csv",
    CLEANED_DIR / "cleaned_understat_2023.csv",
    CLEANED_DIR / "cleaned_understat_2024.csv",
]

# Current 2025/26 PL teams
CURRENT_TEAMS = [
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford",
    "Brighton and Hove Albion", "Burnley", "Chelsea", "Crystal Palace",
    "Everton", "Fulham", "Leeds United", "Liverpool",
    "Manchester City", "Manchester United", "Newcastle United",
    "Nottingham Forest", "Sunderland", "Tottenham Hotspur",
    "West Ham United", "Wolverhampton Wanderers",
]


# -----------------------------------------------------
# Load Prem match-result files (already cleaned)
# -----------------------------------------------------
def load_prem_results() -> pd.DataFrame:
    frames = []
    for f in PREM_FILES:
        if not f.exists():
            print(f"⚠️ Missing Prem file: {f}")
            continue
        df = pd.read_csv(f, parse_dates=["Date"])
        df = df.rename(columns={
            "Date": "date",
            "HomeTeam": "home_team",
            "AwayTeam": "away_team",
            "FTHG": "goals_home",
            "FTAG": "goals_away",
            "FTR": "result",
            "HY": "yellow_home",
            "AY": "yellow_away",
            "HR": "red_home",
            "AR": "red_away"
        })
        frames.append(df)

    prem = pd.concat(frames, ignore_index=True)
    prem["season"] = prem["date"].dt.year
    return prem


# -----------------------------------------------------
# Load Understat xG data
# -----------------------------------------------------
def load_understat_xg() -> pd.DataFrame:
    frames = []
    for f in UNDERSTAT_FILES:
        if not f.exists():
            print(f"⚠️ Missing Understat file: {f}")
            continue

        df = pd.read_csv(f)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])

        df = df.rename(columns={
            "club_name": "team",
            "xG": "xG_for",
            "xGA": "xG_against"
        })

        frames.append(df)

    xg = pd.concat(frames, ignore_index=True)

    return xg[["date", "team", "xG_for", "xG_against"]]


# -----------------------------------------------------
# Build team–match panel
# -----------------------------------------------------
def build_team_match_panel() -> pd.DataFrame:
    prem = load_prem_results()
    xg = load_understat_xg()

    prem_home = prem.rename(columns={
        "home_team": "team",
        "away_team": "opponent",
        "goals_home": "goals_for",
        "goals_away": "goals_against",
        "yellow_home": "yellow",
        "red_home": "red",
    })

    prem_away = prem.rename(columns={
        "away_team": "team",
        "home_team": "opponent",
        "goals_away": "goals_for",
        "goals_home": "goals_against",
        "yellow_away": "yellow",
        "red_away": "red",
    })

    prem_home["home_away"] = "H"
    prem_away["home_away"] = "A"

    panel = pd.concat([prem_home, prem_away], ignore_index=True)
    panel = panel.sort_values("date")

    # Convert form result to points
    panel["pts"] = panel["result"].map({"H": 3, "D": 1, "A": 0})
    panel.loc[panel["home_away"] == "A", "pts"] = panel["result"].map({"A": 3, "D": 1, "H": 0})

    # Merge xG
    panel = panel.merge(
        xg,
        on=["date", "team"],
        how="left"
    )

    panel["xG_for"] = panel["xG_for"].fillna(0)
    panel["xG_against"] = panel["xG_against"].fillna(0)

    return panel


# -----------------------------------------------------
# Add engineered features
# -----------------------------------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["team", "date"]).copy()

    df["xGD"] = df["xG_for"] - df["xG_against"]
    df["clean_sheet"] = (df["goals_against"] == 0).astype(int)
    df["card_points"] = df["yellow"] + 2 * df["red"]

    df["form3"] = df.groupby("team")["pts"].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )

    df["team_avg_pts"] = df.groupby("team")["pts"].transform(
        lambda x: x.expanding().mean()
    )

    opp_avg = (
        df[["date", "team", "team_avg_pts"]]
        .rename(columns={"team": "opponent", "team_avg_pts": "opp_avg_pts"})
    )

    df = df.merge(opp_avg, on=["date", "opponent"], how="left")
    df["opp_avg_pts"] = df["opp_avg_pts"].fillna(df["opp_avg_pts"].mean())

    return df


# -----------------------------------------------------
# Main pipeline
# -----------------------------------------------------
def build_feature_dataset():
    print("Building team–match panel...")
    panel = build_team_match_panel()

    print("Adding engineered features...")
    features = add_features(panel)

    FEATURES_OUT.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(FEATURES_OUT, index=False)
    print(f"✅ Saved feature dataset to: {FEATURES_OUT}")

    return features


if __name__ == "__main__":
    df = build_feature_dataset()
    print(df.head())