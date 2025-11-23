import pandas as pd
from pathlib import Path

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_2019_TABLE = PROJECT_ROOT / "data" / "Cleaned Data" / "2019_table_raw.csv"
OUT_FILE = PROJECT_ROOT / "data" / "Cleaned Data" / "starting_prices_2019.csv"

# ------------------------------------------------------------------
# Generate starting stock prices based on 2019 points
# ------------------------------------------------------------------
def generate_starting_prices():
    """
    Convert 2019 Premier League final points into starting stock prices.
    
    starting_price = min_price + 
                     (points - min_pts) / (max_pts - min_pts) * (max_price - min_price)
                     
    - Strong 2019 teams start higher
    - Weak teams start lower
    - Teams not in the table will later get a default value in pricing.py
    """
    if not RAW_2019_TABLE.exists():
        raise FileNotFoundError(
            f"❌ Missing 2019 table file:\n{RAW_2019_TABLE}\n"
            f"Please create it first with columns: team,points"
        )

    # Load raw 2019 table
    df = pd.read_csv(RAW_2019_TABLE)
    if "team" not in df.columns or "points" not in df.columns:
        raise ValueError("❌ CSV must contain columns: 'team' and 'points'")

    # Normalisation bounds
    min_price = 80
    max_price = 220

    min_pts = df["points"].min()
    max_pts = df["points"].max()

    # Apply normalisation
    df["start_price"] = (
        min_price
        + (df["points"] - min_pts) / (max_pts - min_pts) * (max_price - min_price)
    )

    # Only output required columns
    df_out = df[["team", "start_price"]].copy()

    # Save output
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_FILE, index=False)

    print(f"✅ Successfully generated starting prices:")
    print(df_out)
    print(f"\n📁 Saved to: {OUT_FILE}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == "__main__":
    generate_starting_prices()