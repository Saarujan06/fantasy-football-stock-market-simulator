from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# Core helper: build team-level expected points & uncertainty
# ----------------------------------------------------------------------------

def _build_team_level_metrics(pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Take the future_predictions_2025_26 DataFrame (home-team perspective) and
    build a long-form DataFrame with one row per (team, match), containing:

        - exp_points : expected points for that team in that match
        - uncertainty: 1 - max(prob_win, prob_draw, prob_loss)
                       (higher = more uncertain prediction)
    """
    required_cols = [
        "home_team",
        "away_team",
        "prob_win",
        "prob_draw",
        "prob_loss",
    ]
    missing = [c for c in required_cols if c not in pred_df.columns]
    if missing:
        raise KeyError(
            f"future_predictions is missing required columns: {missing}. "
            "Check predict_future.py output."
        )

    records = []

    for _, row in pred_df.iterrows():
        home = str(row["home_team"])
        away = str(row["away_team"])

        p_win = float(row["prob_win"])
        p_draw = float(row["prob_draw"])
        p_loss = float(row["prob_loss"])

        # Expected points from home-team perspective:
        #   home:  3 * P(win) + 1 * P(draw)
        #   away:  3 * P(away win) + 1 * P(draw)
        #         = 3 * P(home loss) + 1 * P(draw)
        exp_home = 3.0 * p_win + 1.0 * p_draw
        exp_away = 3.0 * p_loss + 1.0 * p_draw

        # Simple uncertainty proxy: how "spread out" the probabilities are.
        # If one outcome is ~certain (p≈1), max_prob≈1 → uncertainty≈0
        # If probabilities are flatter, max_prob is smaller → uncertainty↑
        max_prob = max(p_win, p_draw, p_loss)
        uncertainty = 1.0 - max_prob

        # One row for the home team
        records.append(
            {
                "team": home,
                "exp_points": exp_home,
                "uncertainty": uncertainty,
            }
        )

        # One row for the away team
        records.append(
            {
                "team": away,
                "exp_points": exp_away,
                "uncertainty": uncertainty,
            }
        )

    team_match_df = pd.DataFrame(records)
    return team_match_df


# ----------------------------------------------------------------------------
# Core helper: aggregate to per-team stock signal
# ----------------------------------------------------------------------------

def _aggregate_team_signals(team_match_df: pd.DataFrame) -> pd.DataFrame:
    """
    From the long-form team_match_df (one row per (team, match)), compute:

        - matches            : number of matches used (should be 1 for next GW)
        - total_exp_points   : sum of expected points
        - avg_exp_points     : mean expected points per match
        - avg_uncertainty    : mean prediction uncertainty per match
        - risk_adj_score     : avg_exp_points / (1 + avg_uncertainty)

    Then:
        - rank teams by risk_adj_score (descending)
        - compute percentile positions
        - assign BUY / HOLD / SELL using percentile thresholds
        - compute signal_strength as z-score of risk_adj_score
    """
    if team_match_df.empty:
        raise ValueError("team_match_df is empty; no matches to aggregate.")

    grouped = team_match_df.groupby("team", as_index=False).agg(
        matches=("exp_points", "count"),
        total_exp_points=("exp_points", "sum"),
        avg_exp_points=("exp_points", "mean"),
        avg_uncertainty=("uncertainty", "mean"),
    )

    # Risk-adjusted expected points:
    # Higher expected points is good; higher uncertainty is bad.
    grouped["risk_adj_score"] = grouped["avg_exp_points"] / (
        1.0 + grouped["avg_uncertainty"]
    )

    # Rank: 1 = highest risk-adjusted score
    grouped = grouped.sort_values("risk_adj_score", ascending=False).reset_index(
        drop=True
    )
    grouped["rank"] = np.arange(1, len(grouped) + 1)

    # Percentile of risk_adj_score (0-100)
    scores = grouped["risk_adj_score"]
    p30 = float(scores.quantile(0.30))
    p70 = float(scores.quantile(0.70))

    def classify_signal(score: float) -> str:
        if score >= p70:
            return "BUY"
        elif score <= p30:
            return "SELL"
        else:
            return "HOLD"

    grouped["signal"] = grouped["risk_adj_score"].apply(classify_signal)

    # signal_strength = z-score of risk_adj_score
    mean_score = float(scores.mean())
    std_score = float(scores.std(ddof=0))
    if std_score > 0:
        grouped["signal_strength"] = (grouped["risk_adj_score"] - mean_score) / std_score
    else:
        # All scores identical → no dispersion
        grouped["signal_strength"] = 0.0

    # Also store the percentile itself (for debugging / explanation)
    grouped["percentile"] = scores.rank(pct=True) * 100.0

    return grouped


# ----------------------------------------------------------------------------
# Public API used by main.py
# ----------------------------------------------------------------------------

def compute_stock_directions(
    future_predictions: pd.DataFrame,
    results_dir: str | Path,
) -> pd.DataFrame:
    """
    Entry point used by main.py.

    main.py calls:
        compute_stock_directions(
            future_predictions=future_pred_df,
            results_dir=RESULTS_DIR,
        )

    So here we accept the predictions DataFrame directly, compute
    per-team stock signals, and save stock_direction_2025_26.csv.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if future_predictions is None or future_predictions.empty:
        raise ValueError(
            "future_predictions DataFrame is empty; "
            "run predict_future_matches() first."
        )

    pred_df = future_predictions.copy()

    # 1) Build per-(team,match) expected points & uncertainty
    team_match_df = _build_team_level_metrics(pred_df)

    # 2) Aggregate to per-team stock signals
    stock_df = _aggregate_team_signals(team_match_df)

    # 3) Save
    out_path = results_dir / "stock_direction_2025_26.csv"
    stock_df.to_csv(out_path, index=False)
    print(f"Saved stock direction summary to: {out_path}")

    return stock_df


if __name__ == "__main__":
    # Optional: manual test if you want to run this file directly
    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / "results"
    preds_path = results_dir / "future_predictions_2025_26.csv"

    if preds_path.exists():
        df_preds = pd.read_csv(preds_path)
        compute_stock_directions(df_preds, results_dir)
    else:
        print(f"No predictions file found at {preds_path}")