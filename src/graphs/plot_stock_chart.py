from __future__ import annotations

from pathlib import Path
import pandas as pd
import plotly.graph_objects as go


# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"

PRICE_TS_PATH = RESULTS_DIR / "pricing_engine" / "price_timeseries.csv"
FUTURE_PRED_PATH = RESULTS_DIR / "forecasts" / "future_predictions_2025_26.csv"
OUT_HTML = RESULTS_DIR / "charts" / "next_week_stock_overlay.html"


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    raise KeyError(f"Could not find any of {candidates} in columns: {list(df.columns)}")


def predicted_outcome_and_probs(row: pd.Series, team: str, home_col: str, away_col: str) -> tuple[str, float, float, float]:
    p_win_h = float(row["prob_win"])
    p_draw_h = float(row["prob_draw"])
    p_loss_h = float(row["prob_loss"])

    is_home = str(row[home_col]) == team
    if is_home:
        p_win, p_draw, p_loss = p_win_h, p_draw_h, p_loss_h
    else:
        p_win, p_draw, p_loss = p_loss_h, p_draw_h, p_win_h

    outcome = ["LOSS", "DRAW", "WIN"][int(pd.Series([p_loss, p_draw, p_win]).idxmax())]
    return outcome, p_win, p_draw, p_loss


def outcome_color(outcome: str) -> str:
    return {"WIN": "limegreen", "LOSS": "red", "DRAW": "gold"}[outcome]


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def plot_stock_chart(auto_open: bool = True) -> Path:
    price_df = pd.read_csv(PRICE_TS_PATH)
    pred_df = pd.read_csv(FUTURE_PRED_PATH)

    # ---- find columns ----
    team_col = _find_col(price_df, ["team", "club", "club_name"])
    date_col = _find_col(price_df, ["date", "match_date", "game_date"])
    price_col = _find_col(price_df, ["price_scaled", "price_raw", "price", "stock_price", "current_price", "value"])

    home_col = _find_col(pred_df, ["home_team"])
    away_col = _find_col(pred_df, ["away_team"])
    pred_date_col = _find_col(pred_df, ["date"])

    # ---- parse dates ----
    price_df[date_col] = pd.to_datetime(price_df[date_col], errors="coerce")
    pred_df[pred_date_col] = pd.to_datetime(pred_df[pred_date_col], errors="coerce")

    price_df = price_df.dropna(subset=[team_col, date_col, price_col]).copy()
    pred_df = pred_df.dropna(subset=[home_col, away_col, pred_date_col, "prob_win", "prob_draw", "prob_loss"]).copy()

    teams = sorted(set(pred_df[home_col].astype(str)) | set(pred_df[away_col].astype(str)))
    if not teams:
        raise ValueError("No teams found in future predictions.")

    fig = go.Figure()
    buttons = []

    # We add exactly 2 traces per team: (1) price history, (2) next fixture marker
    for i, team in enumerate(teams):
        team_prices = price_df[price_df[team_col].astype(str) == team].sort_values(date_col)
        if team_prices.empty:
            # keep dropdown alignment anyway
            fig.add_trace(go.Scatter(x=[], y=[], mode="lines", name="Price history", visible=(i == 0)))
            fig.add_trace(go.Scatter(x=[], y=[], mode="markers", name="Next fixture", visible=(i == 0)))
            continue

        # ---- price history ----
        fig.add_trace(
            go.Scatter(
                x=team_prices[date_col],
                y=team_prices[price_col],
                mode="lines+markers",
                line=dict(color="#9aa0a6", width=2),
                marker=dict(size=5),
                name="Price history",
                visible=(i == 0),
                hovertemplate="Date %{x|%b %Y}<br>Price %{y:.2f}<extra></extra>",
            )
        )

        # ---- next fixture marker (with probs in hover) ----
        team_fixtures = pred_df[
            (pred_df[home_col].astype(str) == team) | (pred_df[away_col].astype(str) == team)
        ].sort_values(pred_date_col)

        if team_fixtures.empty:
            fig.add_trace(go.Scatter(x=[], y=[], mode="markers", name="Next fixture", visible=(i == 0)))
        else:
            next_fix = team_fixtures.iloc[0]
            outcome, p_win, p_draw, p_loss = predicted_outcome_and_probs(next_fix, team, home_col, away_col)
            color = outcome_color(outcome)

            latest_price = float(team_prices[price_col].iloc[-1])

            # text shown on chart (keep short)
            label_text = f"{outcome}  ({p_win:.2f}/{p_draw:.2f}/{p_loss:.2f})"

            # simple hover without customdata (string interpolation)
            if str(next_fix[home_col]) == team:
                fixture_str = f"{team} vs {next_fix[away_col]} (H)"
            else:
                fixture_str = f"{team} @ {next_fix[home_col]} (A)"

            hover = (
                f"Date %{{x|%b %Y}}<br>"
                f"Fixture: {fixture_str}<br>"
                f"P(win/draw/loss): {p_win:.3f} / {p_draw:.3f} / {p_loss:.3f}<br>"
                f"Predicted: {outcome}"
                "<extra></extra>"
            )

            fig.add_trace(
                go.Scatter(
                    x=[next_fix[pred_date_col]],
                    y=[latest_price],
                    mode="markers+text",
                    marker=dict(size=16, color=color),
                    text=[label_text],
                    textposition="top center",
                    name="Next fixture (prediction)",
                    visible=(i == 0),
                    hovertemplate=hover,
                )
            )

        # ---- dropdown visibility (2 traces per team) ----
        vis = [False] * (len(teams) * 2)
        vis[i * 2] = True
        vis[i * 2 + 1] = True

        buttons.append(
            dict(
                label=team,
                method="update",
                args=[{"visible": vis}, {"title": f"{team} — Price history + next fixture probabilities"}],
            )
        )

    fig.update_layout(
        template="plotly_dark",
        title=f"{teams[0]} — Price history + next fixture probabilities",
        hovermode="x",
        updatemenus=[dict(type="dropdown", x=0.01, y=1.15, buttons=buttons)],
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=40, r=30, t=60, b=40),
    )

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(OUT_HTML, auto_open=auto_open)
    print(f"Saved interactive chart to {OUT_HTML}")
    return OUT_HTML


if __name__ == "__main__":
    plot_stock_chart()