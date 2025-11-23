from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEAN_DIR = PROJECT_ROOT / "data" / "cleaned_data"
PRICE_FILE = CLEAN_DIR / "team_prices_2025.csv"


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def load_price_data() -> pd.DataFrame:
    if not PRICE_FILE.exists():
        raise FileNotFoundError(f"Price file not found: {PRICE_FILE}")

    df = pd.read_csv(PRICE_FILE, parse_dates=["date"])
    if df.empty:
        raise ValueError("Price file is empty.")
    return df


def plot_team(df_team: pd.DataFrame, team: str):
    plt.figure(figsize=(12, 6))

    # --- NVDA-style visual design ---
    plt.plot(
        df_team["date"],
        df_team["price"],
        color="#0077cc",
        linewidth=2.2,
    )

    # subtle points (optional, not too strong)
    plt.scatter(
        df_team["date"],
        df_team["price"],
        color="#0077cc",
        s=18,
        alpha=0.7
    )

    plt.title(f"Stock Price — {team}", fontsize=20, pad=15)
    plt.xlabel("Date", fontsize=13)
    plt.ylabel("Stock Price (£)", fontsize=13)

    # grid similar to NVDA chart
    plt.grid(
        True,
        linestyle="--",
        linewidth=0.7,
        alpha=0.4
    )

    # smooth x-axis formatting
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))

    plt.tight_layout()
    plt.show()


def show_team(team: str | None = None):
    df = load_price_data()
    teams = sorted(df["team"].unique())

    if team is None:
        print("\nAvailable teams:\n")
        for t in teams:
            print("  -", t)
        print("\nUsage:")
        print("   python3 -m src.app.show_team_price \"Arsenal\"")
        return

    if team not in teams:
        print(f"\nTeam '{team}' not found.")
        print("Available teams:", ", ".join(teams))
        return

    df_team = df[df["team"] == team].sort_values("date")

    plot_team(df_team, team)


# ---------------------------------------------------------
# Entry
# ---------------------------------------------------------
if __name__ == "__main__":
    # team name passed as command-line argument
    team_arg = sys.argv[1] if len(sys.argv) > 1 else None
    show_team(team_arg)