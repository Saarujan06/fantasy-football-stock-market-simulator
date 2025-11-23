from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEANED_DIR = PROJECT_ROOT / "data" / "Cleaned Data"

FEATURES_FILE = CLEANED_DIR / "team_match_features.csv"
PRICES_OUT = CLEANED_DIR / "team_prices.csv"
STARTING_PRICES_FILE = CLEANED_DIR / "starting_prices_2019.csv"

# Current 2025/26 PL teams
CURRENT_TEAMS = [
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford",
    "Brighton and Hove Albion", "Burnley", "Chelsea", "Crystal Palace",
    "Everton", "Fulham", "Leeds United", "Liverpool",
    "Manchester City", "Manchester United", "Newcastle United",
    "Nottingham Forest", "Sunderland", "Tottenham Hotspur",
    "West Ham United", "Wolverhampton Wanderers",
]


# ============================================================
# 1. Load starting prices (based on 2019 table)
# ============================================================
def load_starting_prices() -> dict[str, float]:
    """
    Load team-specific starting prices from starting_prices_2019.csv

    CSV format:
        team,start_price

    If the file doesn't exist, we return an empty dict and later default
    all teams to the same starting price (100).
    """
    if not STARTING_PRICES_FILE.exists():
        print(f"⚠️ No starting price file found at {STARTING_PRICES_FILE}")
        print("   → All teams will start at price 100.")
        return {}

    df = pd.read_csv(STARTING_PRICES_FILE)
    if "team" not in df.columns or "start_price" not in df.columns:
        raise ValueError("starting_prices_2019.csv must contain 'team' and 'start_price' columns")

    df = df.dropna(subset=["team", "start_price"])
    df["start_price"] = df["start_price"].astype(float)

    mapping = dict(zip(df["team"], df["start_price"]))
    print(f"Loaded {len(mapping)} starting prices from 2019 table.")
    return mapping


# ============================================================
# 2. Fit Ridge regression to learn feature weights
# ============================================================
def fit_pricing_coefficients(df: pd.DataFrame) -> dict[str, float]:
    """
    Learn coefficients for the pricing formula using Ridge regression.

    Regression model:
        pts_t ≈ β * xGD_t + γ * clean_sheet_t + δ_raw * card_points_t
                + ε_raw * opp_avg_pts_t + ζ

    Then we use:
        ΔP_t = α * pts_t
               + β * xGD_t
               + γ * clean_sheet_t
               - δ * card_points_t
               - ε * opp_avg_pts_t
               + ζ

    where δ = |δ_raw|, ε = |ε_raw|, and α is a manual positive weight on points.
    """
    feature_cols = ["xGD", "clean_sheet", "card_points", "opp_avg_pts"]

    reg_df = df[feature_cols + ["pts"]].copy()

    # Fill NaNs to avoid dropping rows
    reg_df["xGD"] = reg_df["xGD"].fillna(0.0)
    reg_df["clean_sheet"] = reg_df["clean_sheet"].fillna(0.0)
    reg_df["card_points"] = reg_df["card_points"].fillna(0.0)
    # Opponent strength: fill with mean so we keep structure
    reg_df["opp_avg_pts"] = reg_df["opp_avg_pts"].fillna(reg_df["opp_avg_pts"].mean())
    reg_df["pts"] = reg_df["pts"].fillna(0.0)

    print(f"Regression dataset size: {len(reg_df)} rows")

    if len(reg_df) < 100:
        print("⚠️ Not enough data for regression. Using hand-picked defaults.")
        shrink = 0.15
        return {
            "alpha": 0.4 * shrink,
            "beta": 0.5 * shrink,
            "gamma": 0.3 * shrink,
            "delta": 0.1 * shrink,
            "epsilon": 0.05 * shrink,
            "zeta": 0.2 * shrink,
        }

    X = reg_df[feature_cols].values
    y = reg_df["pts"].values

    # Add intercept column
    X_with_intercept = np.c_[np.ones(len(X)), X]

    # Ridge: θ = (XᵀX + λI)^{-1} Xᵀy, no penalty on intercept
    lambda_ridge = 1.0
    n_params = X_with_intercept.shape[1]

    I = np.eye(n_params)
    I[0, 0] = 0.0  # don't penalise intercept

    XtX = X_with_intercept.T @ X_with_intercept
    Xty = X_with_intercept.T @ y

    theta = np.linalg.solve(XtX + lambda_ridge * I, Xty)

    zeta = theta[0]        # intercept
    beta = theta[1]        # xGD
    gamma = theta[2]       # clean_sheet
    delta_raw = theta[3]   # card_points
    epsilon_raw = theta[4] # opp_avg_pts

    # These will be used as penalties in the stock formula
    delta = abs(delta_raw)
    epsilon = abs(epsilon_raw)

    print("\nRidge regression coefficients (raw):")
    print(f"  ζ (intercept):       {zeta:.4f}")
    print(f"  β (xGD):             {beta:.4f}")
    print(f"  γ (clean_sheet):     {gamma:.4f}")
    print(f"  δ_raw (card_points): {delta_raw:.4f}")
    print(f"  ε_raw (opp_avg_pts): {epsilon_raw:.4f}")

    # Global shrink factor to keep price movements realistic
    shrink = 0.15

    coefs = {
        "alpha": 0.4 * shrink,   # points weight (wins/draws/losses)
        "beta":  beta * shrink,
        "gamma": gamma * shrink,
        "delta": delta * shrink,
        "epsilon": epsilon * shrink,
        "zeta":  zeta * shrink,
    }

    print("\nScaled coefficients for pricing formula:")
    for k, v in coefs.items():
        print(f"  {k}: {v:.4f}")

    return coefs


# ============================================================
# 3. Compute stock prices over time
# ============================================================
def compute_stock_prices(
    df: pd.DataFrame,
    coefs: dict[str, float],
    starting_price_map: dict[str, float]
) -> pd.DataFrame:
    """
    Compute cumulative stock prices for each team:

        ΔP_t = α * pts_t
               + β * xGD_t
               + γ * clean_sheet_t
               - δ * card_points_t
               - ε * opp_avg_pts_t
               + ζ

        P_t = P0_team + Σ_{s ≤ t} ΔP_s

    where P0_team comes from the 2019-based starting_prices file, or
    defaults to a common lower value if the team wasn't present in 2019.
    """
    df = df.sort_values(["team", "date"]).copy()

    # Ensure required columns exist and are filled
    for col in ["pts", "xGD", "clean_sheet", "card_points", "opp_avg_pts"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0)

    # Compute per-match price change
    df["delta_P"] = (
        coefs["alpha"] * df["pts"]
        + coefs["beta"] * df["xGD"]
        + coefs["gamma"] * df["clean_sheet"]
        - coefs["delta"] * df["card_points"]
        - coefs["epsilon"] * df["opp_avg_pts"]
        + coefs["zeta"]
    ).fillna(0.0)

    # Determine default starting price for teams without 2019 info
    if starting_price_map:
        min_start = min(starting_price_map.values())
        default_start = max(min_start - 5.0, 50.0)  # "below the weakest 2019 team", but not crazy low
    else:
        default_start = 100.0

    # Attach per-team starting price
    df["P0"] = df["team"].map(starting_price_map).fillna(default_start)

    # Cumulative sum of delta_P for each team
    df["price"] = df.groupby("team")["delta_P"].cumsum() + df["P0"]

    # Prevent negative prices
    df["price"] = df["price"].clip(lower=1.0)

    return df


# ============================================================
# 4. Main pipeline
# ============================================================
def build_pricing_dataset(save_to_csv: bool = True) -> pd.DataFrame:
    """
    Pipeline:
      1. Load engineered feature dataset (all teams, 2020–2026)
      2. Fit Ridge regression to learn coefficients
      3. Load 2019-based starting prices
      4. Compute stock price paths for all teams
      5. Filter to CURRENT_TEAMS (2025/26 PL)
      6. Optionally save to CSV as team_prices.csv
    """
    if not FEATURES_FILE.exists():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURES_FILE}\n"
            "→ Run 'python3 -m src.calculations.feature_engineering' first."
        )

    # 1. Load features
    df_feat = pd.read_csv(FEATURES_FILE, parse_dates=["date"])
    print(f"Loaded features: {len(df_feat)} rows, {df_feat['team'].nunique()} teams total")

    # 2. Fit coefficients on ALL teams
    print("\n" + "=" * 60)
    print("Fitting pricing coefficients (Ridge regression)...")
    print("=" * 60)
    coefs = fit_pricing_coefficients(df_feat)

    # 3. Load starting prices
    print("\n" + "=" * 60)
    print("Loading starting prices (2019 table)...")
    print("=" * 60)
    starting_price_map = load_starting_prices()

    # 4. Compute prices for ALL teams
    print("\n" + "=" * 60)
    print("Computing stock prices for ALL teams...")
    print("=" * 60)
    priced_all = compute_stock_prices(df_feat, coefs, starting_price_map)

    # 5. Filter to the 20 current 2025/26 teams
    priced = priced_all[priced_all["team"].isin(CURRENT_TEAMS)].copy()
    print(f"\nFiltered to CURRENT_TEAMS: {len(priced)} rows, {priced['team'].nunique()} teams")

    # 6. Save
    if save_to_csv:
        PRICES_OUT.parent.mkdir(parents=True, exist_ok=True)
        priced.to_csv(PRICES_OUT, index=False)
        print(f"\n✅ Saved pricing dataset to: {PRICES_OUT}")

    # Summary
    print("\n" + "=" * 60)
    print("PRICING SUMMARY (CURRENT TEAMS)")
    print("=" * 60)
    final_prices = priced.groupby("team")["price"].last().sort_values(ascending=False)
    print(f"\nFinal price range: £{final_prices.min():.2f} – £{final_prices.max():.2f}")
    print(f"Average final price: £{final_prices.mean():.2f}")
    print("\nTop 5 teams by final price:")
    print(final_prices.head())
    print("\nBottom 5 teams by final price:")
    print(final_prices.tail())

    return priced


if __name__ == "__main__":
    df_prices = build_pricing_dataset(save_to_csv=True)
    print("\n" + "=" * 60)
    print("SAMPLE PRICING DATA")
    print("=" * 60)
    print(df_prices[["date", "team", "opponent", "pts", "xGD", "delta_P", "price"]].head(30))