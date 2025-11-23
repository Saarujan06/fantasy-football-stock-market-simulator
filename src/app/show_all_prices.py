from __future__ import annotations

from pathlib import Path
import math

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEAN_DIR = PROJECT_ROOT / "data" / "cleaned_data"
PRICES_FILE = CLEAN_DIR / "team_prices_2025.csv"

TARGET_SEASON = "2025-2026"


def load_prices() -> pd.DataFrame:
    """
    Load the 2025/26 pricing dataset and filter to the target season.
    """
    if not PRICES_FILE.exists():
        raise FileNotFoundError(f"Price file not found: {PRICES_FILE}")

    df = pd.read_csv(PRICES_FILE, parse_dates=["date"])
    if "season" not in df.columns:
        raise ValueError("Expected a 'season' column in team_prices_2025.csv")

    df = df[df["season"] == TARGET_SEASON].copy()
    if df.empty:
        raise ValueError(f"No rows for season {TARGET_SEASON} in {PRICES_FILE}")

    # Make sure ordering is consistent
    df = df.sort_values(["team", "date"])
    return df


def plot_all_teams_subplots(df: pd.DataFrame) -> None:
    """
    Plot one subplot per team, each with its own y-axis scale (absolute prices).
    This avoids flat lines while keeping true stock-price levels.
    """
    teams = sorted(df["team"].unique())
    n_teams = len(teams)

    # Layout: try 4 columns, adjust rows automatically
    n_cols = 4
    n_rows = math.ceil(n_teams / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 2.5 * n_rows),
        sharex=True,        # same x across all; y is independent
    )

    # If there are fewer teams than grid cells, some axes will be unused
    axes = axes.flatten() if n_teams > 1 else [axes]

    for ax, team in zip(axes, teams):
        team_df = df[df["team"] == team]

        ax.plot(team_df["date"], team_df["price"], linewidth=1.5)

        ax.set_title(team, fontsize=9)
        ax.tick_params(axis="both", labelsize=7)

        # Slight padding around min/max to make it breathe
        y_min = team_df["price"].min()
        y_max = team_df["price"].max()
        padding = 0.03 * (y_max - y_min if y_max > y_min else max(y_min, 1.0))
        ax.set_ylim(y_min - padding, y_max + padding)

        # Light grid for readability
        ax.grid(True, linestyle="--", alpha=0.3)

    # Hide any leftover empty axes
    for j in range(len(teams), len(axes)):
        fig.delaxes(axes[j])

    # Formatting the shared x-axis
    fig.suptitle("Team Stock Prices — 2025/26 Season", fontsize=16, y=0.98)
    fig.autofmt_xdate(rotation=45)
    for ax in axes[-n_cols:]:  # bottom row: keep x labels
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def main() -> None:
    df = load_prices()
    plot_all_teams_subplots(df)


if __name__ == "__main__":
    main()