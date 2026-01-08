from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.team_names import CANONICAL_TEAMS_2526

# ----------------------------------------------------------------------------
# Paths (self-contained outputs)
# ----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
FORECASTS_DIR = RESULTS_DIR / "forecasts"

DEFAULT_FILENAME = "stock_direction_2025_26.csv"


# ----------------------------------------------------------------------------
# Core helper: build team-level expected points & uncertainty
# ----------------------------------------------------------------------------

def _build_team_level_metrics(pred_df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["home_team", "away_team", "prob_win", "prob_draw", "prob_loss"]
    missing = [c for c in required_cols if c not in pred_df.columns]
    if missing:
        raise KeyError(
            f"future_predictions is missing required columns: {missing}. "
            "Check predict_future.py output."
        )

    records: list[dict[str, object]] = []

    for _, row in pred_df.iterrows():
        home = str(row["home_team"])
        away = str(row["away_team"])

        p_win = float(row["prob_win"])
        p_draw = float(row["prob_draw"])
        p_loss = float(row["prob_loss"])

        # Expected points from home-team perspective:
        exp_home = 3.0 * p_win + 1.0 * p_draw
        exp_away = 3.0 * p_loss + 1.0 * p_draw

        uncertainty = 1.0 - max(p_win, p_draw, p_loss)

        records.append({"team": home, "exp_points": exp_home, "uncertainty": uncertainty})
        records.append({"team": away, "exp_points": exp_away, "uncertainty": uncertainty})

    return pd.DataFrame(records)


# ----------------------------------------------------------------------------
# Core helper: aggregate to per-team stock signal
# ----------------------------------------------------------------------------

def _aggregate_team_signals(team_match_df: pd.DataFrame) -> pd.DataFrame:
    if team_match_df.empty:
        raise ValueError("team_match_df is empty; no matches to aggregate.")

    grouped = team_match_df.groupby("team", as_index=False).agg(
        matches=("exp_points", "count"),
        total_exp_points=("exp_points", "sum"),
        avg_exp_points=("exp_points", "mean"),
        avg_uncertainty=("uncertainty", "mean"),
    )

    grouped["risk_adj_score"] = grouped["avg_exp_points"] / (1.0 + grouped["avg_uncertainty"])
    grouped = grouped.sort_values("risk_adj_score", ascending=False).reset_index(drop=True)
    grouped["rank"] = np.arange(1, len(grouped) + 1)

    scores = grouped["risk_adj_score"]
    p30 = float(scores.quantile(0.30))
    p70 = float(scores.quantile(0.70))

    def classify_signal(score: float) -> str:
        if score >= p70:
            return "BUY"
        if score <= p30:
            return "SELL"
        return "HOLD"

    grouped["signal"] = grouped["risk_adj_score"].apply(classify_signal)

    mean_score = float(scores.mean())
    std_score = float(scores.std(ddof=0))
    grouped["signal_strength"] = (scores - mean_score) / std_score if std_score > 0 else 0.0
    grouped["percentile"] = scores.rank(pct=True) * 100.0

    return grouped


# ----------------------------------------------------------------------------
# Public API used by main.py
# ----------------------------------------------------------------------------

def compute_stock_directions(
    future_predictions: pd.DataFrame,
    filename: str = DEFAULT_FILENAME,
    restrict_to_current_pl: bool = True,
) -> pd.DataFrame:
    FORECASTS_DIR.mkdir(parents=True, exist_ok=True)

    if future_predictions is None or future_predictions.empty:
        raise ValueError("future_predictions DataFrame is empty; run predict_future_matches() first.")

    pred_df = future_predictions.copy()

    if restrict_to_current_pl:
        pred_df = pred_df[
            pred_df["home_team"].isin(CANONICAL_TEAMS_2526)
            & pred_df["away_team"].isin(CANONICAL_TEAMS_2526)
        ].copy()

    team_match_df = _build_team_level_metrics(pred_df)

    if restrict_to_current_pl:
        team_match_df = team_match_df[team_match_df["team"].isin(CANONICAL_TEAMS_2526)].copy()

    stock_df = _aggregate_team_signals(team_match_df)

    out_path = FORECASTS_DIR / filename
    stock_df.to_csv(out_path, index=False)
    print(f"Saved stock direction summary to: {out_path}")

    return stock_df


if __name__ == "__main__":
    preds_path = FORECASTS_DIR / "future_predictions_2025_26.csv"

    if preds_path.exists():
        df_preds = pd.read_csv(preds_path)
        compute_stock_directions(df_preds)
    else:
        print(f"No predictions file found at {preds_path}")