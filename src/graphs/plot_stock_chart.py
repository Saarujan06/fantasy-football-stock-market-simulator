from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# pip install plotly kaleido
import plotly.graph_objects as go


# ----------------------------------------------------------------------------
# Default paths (follow your results/ folder structure)
# ----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"

DEFAULT_PRICING_DIR = RESULTS_DIR / "pricing_engine"
DEFAULT_FORECASTS_DIR = RESULTS_DIR / "forecasts"
DEFAULT_CHARTS_DIR = RESULTS_DIR / "charts"

PRICE_TS_PATH = DEFAULT_PRICING_DIR / "price_timeseries.csv"
FUTURE_PRED_PATH = DEFAULT_FORECASTS_DIR / "future_predictions_2025_26.csv"
OUT_HTML = DEFAULT_CHARTS_DIR / "next_week_stock_overlay.html"


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _find_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    if required:
        raise KeyError(f"Could not find any of columns {candidates} in: {list(df.columns)}")
    return None


def _coerce_datetime(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce")
    if dt.notna().sum() == 0:
        dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    return dt


# ----------------------------------------------------------------------------
# Public API (called by main.py)
# ----------------------------------------------------------------------------

def plot_stock_chart(
    pricing_dir: Path | None = None,
    forecasts_dir: Path | None = None,
    charts_dir: Path | None = None,
    price_ts_path: Path | None = None,
    future_pred_path: Path | None = None,
    out_html: Path | None = None,
    auto_open: bool = True,
) -> Path:
    """
    Builds an interactive Plotly chart (dropdown by team) overlaying:
      - stock price time series (price_timeseries.csv)
      - next matchweek fixtures + predicted probabilities (future_predictions_2025_26.csv)

    Reads from:
      - results/pricing_engine/price_timeseries.csv
      - results/forecasts/future_predictions_2025_26.csv

    Saves to:
      - results/charts/next_week_stock_overlay.html
    """
    pricing_dir = pricing_dir or DEFAULT_PRICING_DIR
    forecasts_dir = forecasts_dir or DEFAULT_FORECASTS_DIR
    charts_dir = charts_dir or DEFAULT_CHARTS_DIR

    price_ts_path = price_ts_path or (pricing_dir / "price_timeseries.csv")
    future_pred_path = future_pred_path or (forecasts_dir / "future_predictions_2025_26.csv")
    out_html = out_html or (charts_dir / "next_week_stock_overlay.html")

    if not price_ts_path.exists():
        raise FileNotFoundError(f"Missing: {price_ts_path}")
    if not future_pred_path.exists():
        raise FileNotFoundError(f"Missing: {future_pred_path}")

    # -----------------------
    # Load future predictions
    # -----------------------
    pred_df = pd.read_csv(future_pred_path)

    pred_date_col = _find_col(pred_df, ["date"])
    home_col = _find_col(pred_df, ["home_team"])
    away_col = _find_col(pred_df, ["away_team"])
    pw_col = _find_col(pred_df, ["prob_win"])
    pd_col = _find_col(pred_df, ["prob_draw"])
    pl_col = _find_col(pred_df, ["prob_loss"])

    pred_df["__date"] = _coerce_datetime(pred_df[pred_date_col]).dt.normalize()
    pred_df = pred_df.dropna(subset=["__date", home_col, away_col, pw_col, pd_col, pl_col]).copy()
    pred_df["__date_int"] = pred_df["__date"].astype("int64")

    pred_df["home_exp_pts"] = 3.0 * pred_df[pw_col].astype(float) + 1.0 * pred_df[pd_col].astype(float)
    pred_df["away_exp_pts"] = 3.0 * pred_df[pl_col].astype(float) + 1.0 * pred_df[pd_col].astype(float)

    # Only teams that appear in next matchweek predictions
    current_teams = sorted(
        set(pred_df[home_col].astype(str).unique())
        | set(pred_df[away_col].astype(str).unique())
    )
    if not current_teams:
        raise ValueError(
            "No teams found in future predictions. "
            "Check results/forecasts/future_predictions_2025_26.csv."
        )

    # -----------------------
    # Load price time series
    # -----------------------
    price_df = pd.read_csv(price_ts_path)

    team_col = _find_col(price_df, ["team", "club", "club_name"])
    price_col = _find_col(price_df, ["price_scaled", "price_raw", "price", "stock_price", "current_price", "value"])
    date_col = _find_col(price_df, ["date", "match_date", "game_date"], required=False)
    mw_col = _find_col(price_df, ["matchweek", "gw", "round", "round_number"], required=False)

    if date_col is None and mw_col is None:
        raise KeyError("price_timeseries.csv must have either a date column OR a matchweek/round column.")

    if date_col is not None:
        price_df["__x"] = _coerce_datetime(price_df[date_col])
        price_df = price_df.dropna(subset=["__x"]).copy()
        price_df["__x_int"] = price_df["__x"].astype("int64")
        x_for_interp = "__x_int"
        x_plot = "__x"
    else:
        price_df["__x"] = pd.to_numeric(price_df[mw_col], errors="coerce")
        price_df = price_df.dropna(subset=["__x"]).copy()
        price_df["__x_int"] = price_df["__x"]
        x_for_interp = "__x_int"
        x_plot = "__x"

    price_df = price_df.dropna(subset=[team_col, price_col]).copy()
    price_df = price_df[price_df[team_col].astype(str).isin(current_teams)].copy()
    price_df = price_df.sort_values([team_col, "__x_int"])

    teams = current_teams

    fig = go.Figure()
    buttons = []
    traces_per_team = 4

    for t_idx, team in enumerate(teams):
        team_prices = price_df[price_df[team_col].astype(str) == team].copy()
        team_preds_home = pred_df[pred_df[home_col].astype(str) == team].copy()
        team_preds_away = pred_df[pred_df[away_col].astype(str) == team].copy()

        # Price line
        fig.add_trace(
            go.Scatter(
                x=team_prices[x_plot],
                y=team_prices[price_col],
                mode="lines+markers",
                name="Price (this season)",
                visible=(t_idx == 0),
            )
        )

        # Marker y-value helper (interpolate onto price line if dates exist)
        def marker_y(pred_dates_int: np.ndarray) -> np.ndarray:
            if len(team_prices) == 0:
                return np.full(len(pred_dates_int), np.nan, dtype=float)

            last_price = float(team_prices[price_col].iloc[-1])

            if date_col is not None and len(team_prices) >= 2:
                xs = team_prices[x_for_interp].to_numpy(dtype=np.int64)
                ys = team_prices[price_col].to_numpy(dtype=float)
                return np.interp(pred_dates_int, xs, ys, left=last_price, right=last_price)

            return np.full(len(pred_dates_int), last_price, dtype=float)

        # Home markers
        fig.add_trace(
            go.Scatter(
                x=team_preds_home["__date"],
                y=marker_y(team_preds_home["__date_int"].to_numpy(dtype=np.int64)),
                mode="markers+text",
                text=[f"vs {opp} (H)" for opp in team_preds_home[away_col].astype(str)],
                textposition="top center",
                name="Next GW fixtures (home)",
                customdata=(
                    np.stack(
                        [
                            team_preds_home[away_col].astype(str),
                            team_preds_home[pw_col].astype(float),
                            team_preds_home[pd_col].astype(float),
                            team_preds_home[pl_col].astype(float),
                            team_preds_home["home_exp_pts"].astype(float),
                        ],
                        axis=1,
                    )
                    if len(team_preds_home)
                    else None
                ),
                hovertemplate=(
                    "Date %{x}<br>"
                    "Fixture: " + team + " vs %{customdata[0]} (H)<br>"
                    "P(win/draw/loss): %{customdata[1]:.3f} / %{customdata[2]:.3f} / %{customdata[3]:.3f}<br>"
                    "Home exp pts: %{customdata[4]:.2f}"
                    "<extra></extra>"
                ),
                visible=(t_idx == 0),
            )
        )

        # Away markers
        fig.add_trace(
            go.Scatter(
                x=team_preds_away["__date"],
                y=marker_y(team_preds_away["__date_int"].to_numpy(dtype=np.int64)),
                mode="markers+text",
                text=[f"@ {opp} (A)" for opp in team_preds_away[home_col].astype(str)],
                textposition="top center",
                name="Next GW fixtures (away)",
                customdata=(
                    np.stack(
                        [
                            team_preds_away[home_col].astype(str),
                            team_preds_away[pw_col].astype(float),
                            team_preds_away[pd_col].astype(float),
                            team_preds_away[pl_col].astype(float),
                            team_preds_away["away_exp_pts"].astype(float),
                        ],
                        axis=1,
                    )
                    if len(team_preds_away)
                    else None
                ),
                hovertemplate=(
                    "Date %{x}<br>"
                    "Fixture: " + team + " @ %{customdata[0]} (A)<br>"
                    "P(win/draw/loss) from HOME model: %{customdata[1]:.3f} / %{customdata[2]:.3f} / %{customdata[3]:.3f}<br>"
                    "Away exp pts: %{customdata[4]:.2f}"
                    "<extra></extra>"
                ),
                visible=(t_idx == 0),
            )
        )

        # Projection trace (empty placeholder)
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                line=dict(dash="dot", width=2),
                name="Projection (optional)",
                visible=(t_idx == 0),
            )
        )

        # Dropdown visibility mask
        vis = [False] * (len(teams) * traces_per_team)
        base = t_idx * traces_per_team
        for j in range(traces_per_team):
            vis[base + j] = True

        buttons.append(
            dict(
                label=team,
                method="update",
                args=[{"visible": vis}, {"title": f"{team} — Stock Price + Next GW Predictions"}],
            )
        )

    fig.update_layout(
        title=f"{teams[0]} — Stock Price + Next GW Predictions",
        template="plotly_dark",
        margin=dict(l=40, r=30, t=50, b=40),
        hovermode="x unified",
        updatemenus=[dict(type="dropdown", direction="down", x=0.01, y=1.15, buttons=buttons)],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_html, include_plotlyjs="cdn", auto_open=auto_open)
    print(f"Saved interactive chart: {out_html}")
    return out_html


if __name__ == "__main__":
    plot_stock_chart()