from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
from datetime import timedelta

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CLEANED_FEATURES_PATH = DATA_DIR / "cleaned_data" / "team_match_features.csv"
FUTURE_FIXTURES_PATH = DATA_DIR / "raw" / "epl_2025_gmt_standard_time.csv"

# ----------------------------------------------------------------------------
# Team name normalisation (must match build_features.py)
# ----------------------------------------------------------------------------

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
    "Bournemouth": "Bournemouth",
    "Sheffield Utd": "Sheffield United",
    "Sheffield United": "Sheffield United",
    "West Ham United": "West Ham",
    "West Ham": "West Ham",
}


def _normalize_team_name(name: str) -> str:
    if pd.isna(name):
        return name
    return TEAM_NAME_MAP.get(str(name), str(name))


# ----------------------------------------------------------------------------
# Load latest team features from the historical dataset
# ----------------------------------------------------------------------------

def _load_team_latest_features(
    feature_cols: List[str],
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    if not CLEANED_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned features not found at {CLEANED_FEATURES_PATH}. "
            "Run `python -m src.pipeline.build_features` first."
        )

    df = pd.read_csv(CLEANED_FEATURES_PATH)

    for col in ["team", "season", "date"]:
        if col not in df.columns:
            raise KeyError(
                "Expected 'team', 'season', and 'date' columns in cleaned dataset. "
                "Check build_features.py."
            )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df_sorted = df.sort_values(["team", "season", "date"])
    latest_rows = df_sorted.groupby("team").tail(1)

    latest_team_feats: Dict[str, np.ndarray] = {}
    for _, row in latest_rows.iterrows():
        team = str(row["team"])
        latest_team_feats[team] = row[feature_cols].to_numpy(dtype=float)

    global_mean = df[feature_cols].mean(axis=0).to_numpy(dtype=float)
    return latest_team_feats, global_mean


# ----------------------------------------------------------------------------
# Load 2025–26 fixtures
# ----------------------------------------------------------------------------

def _load_fixtures_2025_26() -> pd.DataFrame:
    if not FUTURE_FIXTURES_PATH.exists():
        raise FileNotFoundError(f"Future fixtures file not found at {FUTURE_FIXTURES_PATH}")

    df = pd.read_csv(FUTURE_FIXTURES_PATH)

    if "Date" not in df.columns:
        raise KeyError("Expected 'Date' column in fixture file.")

    dt = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df["date"] = dt.dt.date

    # Home / away team names
    if "HomeTeam" in df.columns:
        df["home_team"] = df["HomeTeam"].apply(_normalize_team_name)
    elif "Home Team" in df.columns:
        df["home_team"] = df["Home Team"].apply(_normalize_team_name)
    else:
        raise KeyError("Expected 'HomeTeam' or 'Home Team' in fixture file.")

    if "AwayTeam" in df.columns:
        df["away_team"] = df["AwayTeam"].apply(_normalize_team_name)
    elif "Away Team" in df.columns:
        df["away_team"] = df["Away Team"].apply(_normalize_team_name)
    else:
        raise KeyError("Expected 'AwayTeam' or 'Away Team' in fixture file.")

    # Keep only valid rows
    df = df.dropna(subset=["date", "home_team", "away_team"])

    # Ensure odds columns exist (can be NaN)
    for col in ["B365H", "B365D", "B365A"]:
        if col not in df.columns:
            df[col] = np.nan

    return df


# ----------------------------------------------------------------------------
# Betting implied probabilities
# ----------------------------------------------------------------------------

def _safe_inverse_odds(odds: pd.Series) -> np.ndarray:
    odds = pd.to_numeric(odds, errors="coerce").to_numpy()
    return np.where(odds > 0, 1.0 / odds, np.nan)


def _add_betting_probabilities(fixtures: pd.DataFrame) -> pd.DataFrame:
    df = fixtures.copy()

    h = _safe_inverse_odds(df["B365H"])
    d = _safe_inverse_odds(df["B365D"])
    a = _safe_inverse_odds(df["B365A"])

    total = h + d + a
    df["b365_home_prob"] = h / total
    df["b365_draw_prob"] = d / total
    df["b365_away_prob"] = a / total

    return df


# ----------------------------------------------------------------------------
# Build future feature matrix (home-team perspective)
# ----------------------------------------------------------------------------

def _build_future_feature_matrix(
    fixtures: pd.DataFrame,
    latest_team_feats: Dict[str, np.ndarray],
    global_mean: np.ndarray,
    feature_cols: List[str],
) -> Tuple[np.ndarray, pd.DataFrame]:
    feature_index = {c: i for i, c in enumerate(feature_cols)}

    rows: List[np.ndarray] = []
    meta_rows: List[Dict[str, object]] = []

    for _, row in fixtures.iterrows():
        home = str(row["home_team"])
        away = str(row["away_team"])

        base_home = latest_team_feats.get(home, global_mean).copy()

        # Override betting-related features if they exist in FEATURE_COLS
        def set_if_present(col_name: str, value) -> None:
            idx = feature_index.get(col_name)
            if idx is not None and not pd.isna(value):
                base_home[idx] = float(value)

        # Raw odds (used for training, but we won't export them)
        set_if_present("B365H", row.get("B365H", np.nan))
        set_if_present("B365D", row.get("B365D", np.nan))
        set_if_present("B365A", row.get("B365A", np.nan))

        # Implied probabilities
        set_if_present("b365_home_prob", row.get("b365_home_prob", np.nan))
        set_if_present("b365_draw_prob", row.get("b365_draw_prob", np.nan))
        set_if_present("b365_away_prob", row.get("b365_away_prob", np.nan))

        if "b365_team_win_prob" in feature_index:
            # home-team perspective => team win prob = home win prob
            set_if_present("b365_team_win_prob", row.get("b365_home_prob", np.nan))

        rows.append(base_home)

        # Minimal metadata ONLY (no xG, no raw odds)
        meta_rows.append(
            {
                "date": row["date"],
                "home_team": home,
                "away_team": away,
            }
        )

    X_future = np.vstack(rows)
    meta_df = pd.DataFrame(meta_rows)
    return X_future, meta_df


# ----------------------------------------------------------------------------
# Public API used by main.py
# ----------------------------------------------------------------------------

def predict_future_matches(
    model,
    feature_cols: List[str],
    results_dir: str | Path,
) -> pd.DataFrame:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if not CLEANED_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned features not found at {CLEANED_FEATURES_PATH}. "
            "Run `python -m src.pipeline.build_features` first."
        )

    df_hist = pd.read_csv(CLEANED_FEATURES_PATH)
    if "date" not in df_hist.columns:
        raise KeyError("Expected 'date' column in cleaned dataset. Check build_features.py.")

    df_hist["date"] = pd.to_datetime(df_hist["date"], errors="coerce")
    last_played_ts = df_hist["date"].max()
    if pd.isna(last_played_ts):
        raise ValueError("Could not infer last played match date from historical data.")

    last_played_date = last_played_ts.date()
    print(f"Last completed match was on {last_played_date}")

    latest_team_feats, global_mean = _load_team_latest_features(feature_cols)

    fixtures = _load_fixtures_2025_26()
    fixtures = _add_betting_probabilities(fixtures)

    future = fixtures[fixtures["date"] > last_played_date].copy()
    if future.empty:
        raise ValueError(
            "No future fixtures available after the last played match date "
            f"({last_played_date})."
        )

    next_date = min(future["date"])
    cutoff = next_date + timedelta(days=6)

    fixtures_next = future[(future["date"] >= next_date) & (future["date"] <= cutoff)].copy()
    if fixtures_next.empty:
        raise ValueError(f"No fixtures found in [{next_date} .. {cutoff}].")

    print(f"Predicting NEXT matchweek automatically: {next_date} → {cutoff} ({len(fixtures_next)} fixtures)")

    X_future, meta_df = _build_future_feature_matrix(
        fixtures_next,
        latest_team_feats,
        global_mean,
        feature_cols,
    )

    proba = model.predict_proba(X_future)
    if not hasattr(model, "classes_"):
        raise AttributeError("Model must have .classes_ attribute for predict_proba.")

    classes = np.array(model.classes_)

    def idx_for_class(c: int) -> int:
        if c not in classes:
            raise ValueError(f"Expected class {c} in model.classes_, got {classes}")
        return int(np.where(classes == c)[0][0])

    idx_loss = idx_for_class(0)
    idx_draw = idx_for_class(1)
    idx_win = idx_for_class(2)

    pred_df = meta_df.copy()
    pred_df["prob_win"] = proba[:, idx_win]
    pred_df["prob_draw"] = proba[:, idx_draw]
    pred_df["prob_loss"] = proba[:, idx_loss]

    pred_df["match_id"] = np.arange(1, len(pred_df) + 1)

    out_path = results_dir / "future_predictions_2025_26.csv"
    pred_df.to_csv(out_path, index=False)
    print(f"Saved future predictions to: {out_path}")

    return pred_df