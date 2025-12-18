from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------
# Default paths
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"

FORECASTS_DIR = RESULTS_DIR / "forecasts"
CHARTS_DIR = RESULTS_DIR / "charts"

INPUT_CSV = FORECASTS_DIR / "stock_direction_2025_26.csv"
OUT_PDF = CHARTS_DIR / "stock_signal_ranking.pdf"


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
def plot_stock_signals(
    input_csv: Path | None = None,
    out_pdf: Path | None = None,
    top_n: int | None = None,
) -> Path:
    """
    Create a PDF ranking chart of stock signals based on risk-adjusted score.

    Reads:
        results/forecasts/stock_direction_2025_26.csv

    Writes:
        results/charts/stock_signal_ranking.pdf
    """
    input_csv = input_csv or INPUT_CSV
    out_pdf = out_pdf or OUT_PDF

    if not input_csv.exists():
        raise FileNotFoundError(f"Missing stock direction file: {input_csv}")

    df = pd.read_csv(input_csv)

    required = {"team", "risk_adj_score", "signal"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"stock_direction CSV missing columns: {sorted(missing)}")

    # Sort best → worst
    df = df.sort_values("risk_adj_score", ascending=False).reset_index(drop=True)

    # Optional: only show top N teams
    if top_n is not None:
        df = df.head(int(top_n)).copy()

    color_map = {
        "BUY": "green",
        "HOLD": "gray",
        "SELL": "red",
    }
    colors = df["signal"].map(color_map).fillna("gray")

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, max(6, 0.35 * len(df))))
    plt.barh(df["team"], df["risk_adj_score"], color=colors)
    plt.gca().invert_yaxis()

    plt.xlabel("Risk-adjusted score")
    plt.ylabel("Team")
    plt.title("Stock Direction Signals — Ranking by Risk-Adjusted Score")

    for i, v in enumerate(df["risk_adj_score"]):
        plt.text(v, i, f" {v:.2f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close()

    print(f"Saved stock signal ranking chart → {out_pdf}")
    return out_pdf


if __name__ == "__main__":
    plot_stock_signals()