from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

# -----------------------------------------------------
#  Paths
# -----------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_FILE = PROJECT_ROOT / "data" / "Cleaned Data" / "team_match_features.csv"
PRICING_OUT = PROJECT_ROOT / "data" / "Cleaned Data" / "team_prices.csv"

# Current 2025/26 Premier League teams
CURRENT_TEAMS = [
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford",
    "Brighton and Hove Albion", "Burnley", "Chelsea", "Crystal Palace",
    "Everton", "Fulham", "Leeds United", "Liverpool",
    "Manchester City", "Manchester United", "Newcastle United",
    "Nottingham Forest", "Sunderland", "Tottenham Hotspur",
    "West Ham United", "Wolverhampton Wanderers",
]

# -----------------------------------------------------
# 1. Fit Ridge regression to learn coefficients
# -----------------------------------------------------
def fit_pricing_coefficients(df: pd.DataFrame):
    """
    Fit a L2-regularised linear regression (Ridge) to predict points from features:
        pts_t ≈ β*xGD_t + γ*clean_sheet_t + δ_raw*card_points_t + ε_raw*opp_avg_pts_t + ζ

    We then transform δ_raw, ε_raw into negative effects in the pricing formula.

    Objective:
        min_θ ||y - Xθ||^2 + λ ||θ||^2   (no penalty on intercept)
    """
    feature_cols = ["xGD", "clean_sheet", "card_points", "opp_avg_pts"]

    # Work on a copy; fill NaNs to avoid dropping data
    reg_df = df[feature_cols + ["pts"]].copy()
    reg_df = reg_df.fillna(0.0)

    print(f"Regression dataset: {len(reg_df)} rows")

    if len(reg_df) < 200:
        print("⚠️ WARNING: Not enough data for regression. Using default (hand-picked) coefficients.")
        shrink = 0.15
        return {
            "alpha": 0.3 * shrink,
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

    # Ridge: (X'X + λI)^{-1} X'y, with no penalty on intercept
    lambda_ridge = 1.0  # you can experiment with 0.1, 1.0, 5.0, etc.
    n_params = X_with_intercept.shape[1]

    I = np.eye(n_params)
    I[0, 0] = 0.0  # do NOT penalise intercept

    XtX = X_with_intercept.T @ X_with_intercept
    Xty = X_with_intercept.T @ y

    theta = np.linalg.solve(XtX + lambda_ridge * I, Xty)

    zeta = theta[0]        # intercept
    beta = theta[1]        # xGD
    gamma = theta[2]       # clean_sheet
    delta_raw = theta[3]   # card_points
    epsilon_raw = theta[4] # opp_avg_pts

    # Make card_points and opp_avg_pts penalising in the price formula
    delta = abs(delta_raw)
    epsilon = abs(epsilon_raw)

    print("\nRidge regression coefficients (raw):")
    print(f"  ζ (intercept):      {zeta:.4f}")
    print(f"  β (xGD):            {beta:.4f}")
    print(f"  γ (clean_sheet):    {gamma:.4f}")
    print(f"  δ_raw (cards):      {delta_raw:.4f}")
    print(f"  ε_raw (opp_avg_pts):{epsilon_raw:.4f}")

    # Global shrink factor to control price volatility
    shrink = 0.15

    coefs = {
        "alpha": 0.3 * shrink,   # manual weight on realised points
        "beta": beta * shrink,
        "gamma": gamma * shrink,
        "delta": delta * shrink,
        "epsilon": epsilon * shrink,
        "zeta": zeta * shrink,
    }

    print("\nScaled coefficients for pricing formula:")
    for k, v in coefs.items():
        print(f"  {k}: {v:.4f}")

    return coefs

# -----------------------------------------------------
# 2. Apply the pricing model to compute stock price over time
# -----------------------------------------------------
def compute_stock_prices(df: pd.DataFrame, coef: dict) -> pd.DataFrame:
    """
    Compute cumulative stock prices for each team based on:
        ΔP_t = α*pts + β*xGD + γ*clean_sheet - δ*card_points - ε*opp_avg_pts + ζ
        P_t = P_{t-1} + ΔP_t, with P_0 = 100 at each team's first match (IPO).
    """
    df = df.sort_values(["team", "date"]).copy()

    # Fill NaNs in used columns
    for col in ["pts", "xGD", "clean_sheet", "card_points", "opp_avg_pts"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0)

    P0 = 100.0

    df["delta_P"] = (
        coef["alpha"] * df["pts"]
        + coef["beta"] * df["xGD"]
        + coef["gamma"] * df["clean_sheet"]
        - coef["delta"] * df["card_points"]
        - coef["epsilon"] * df["opp_avg_pts"]
        + coef["zeta"]
    ).fillna(0.0)

    df["price"] = df.groupby("team")["delta_P"].cumsum() + P0

    # Prevent negative prices
    df["price"] = df["price"].clip(lower=1.0)

    return df

# -----------------------------------------------------
# 3. Pipeline wrapper
# -----------------------------------------------------
def build_pricing_dataset(save_to_csv: bool = True) -> pd.DataFrame:
    """
    1. Load feature dataset (all teams, all seasons in features file)
    2. Fit Ridge regression on all teams
    3. Compute prices for all teams
    4. Filter to CURRENT_TEAMS for reporting
    5. Optionally save to CSV
    """
    print("Loading engineered features...")

    if not FEATURES_FILE.exists():
        raise FileNotFoundError(
            f"Features file not found: {FEATURES_FILE}\n"
            "Run 'python3 -m src.calculations.feature_engineering' first!"
        )

    df = pd.read_csv(FEATURES_FILE, parse_dates=["date"])
    print(f"Loaded {len(df)} rows, {len(df['team'].unique())} teams total")

    print("\n" + "="*60)
    print("Fitting pricing coefficients via Ridge regression (ALL teams)...")
    print("="*60)
    coefs = fit_pricing_coefficients(df)

    print("\n" + "="*60)
    print("Computing stock prices for ALL teams...")
    print("="*60)
    priced_all = compute_stock_prices(df, coefs)

    # Filter to current PL teams
    priced = priced_all[priced_all["team"].isin(CURRENT_TEAMS)].copy()
    print(f"\nFiltered to current PL teams: {len(priced)} rows, {len(priced['team'].unique())} teams")

    # IPO check
    ipo_info = (
        priced.sort_values("date")
              .groupby("team")
              .agg(first_date=("date", "min"),
                   first_price=("price", "first"))
    )
    print("\nIPO CHECK – first match date and IPO price per team:")
    print(ipo_info)

    if save_to_csv:
        PRICING_OUT.parent.mkdir(parents=True, exist_ok=True)
        priced.to_csv(PRICING_OUT, index=False)
        print(f"\n✅ Saved pricing dataset to: {PRICING_OUT}")

    # Summary stats
    print("\n" + "="*60)
    print("PRICING SUMMARY")
    print("="*60)
    final_prices = priced.groupby("team")["price"].last().sort_values(ascending=False)
    print(f"\nPrice range: £{final_prices.min():.2f} - £{final_prices.max():.2f}")
    print(f"Average price: £{final_prices.mean():.2f}")
    print("\nTop 5 teams by final price:")
    print(final_prices.head())
    print("\nBottom 5 teams by final price:")
    print(final_prices.tail())

    return priced


if __name__ == "__main__":
    df_prices = build_pricing_dataset(save_to_csv=True)
    print("\n" + "="*60)
    print("SAMPLE PRICING DATA")
    print("="*60)
    print(df_prices[["date", "team", "opponent", "pts", "xGD", "delta_P", "price"]].head(30))