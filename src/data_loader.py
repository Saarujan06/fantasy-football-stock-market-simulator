from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================================================
# PATHS & GLOBALS
# ============================================================================

# This file lives in src/data_loader.py
# parents[0] = .../src
# parents[1] = project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CLEANED_FEATURES_PATH = DATA_DIR / "cleaned_data" / "team_match_features.csv"


# ============================================================================
# CORE LOADERS
# ============================================================================

def load_features() -> pd.DataFrame:
    """
    Load the cleaned, engineered team-level dataset produced by build_features.py.

    Expected:
        - File: data/cleaned_data/team_match_features.csv
        - Column 'label' with values {0,1,2}
        - No NaNs in any numeric feature columns (guaranteed by build_features)
    """
    if not CLEANED_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned features not found at {CLEANED_FEATURES_PATH}.\n"
            "Make sure you have run: python -m src.pipeline.build_features"
        )

    df = pd.read_csv(CLEANED_FEATURES_PATH)

    if "label" not in df.columns:
        raise KeyError(
            "'label' column not found in cleaned dataset. "
            "Check build_features.py output."
        )

    if df["label"].isna().any():
        raise ValueError("Found NaNs in 'label' column; this should not happen.")
    if df.shape[0] == 0:
        raise ValueError("Loaded dataset is empty.")

    return df


def infer_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Infer which columns to use as features:
      - Take all numeric columns
      - Drop the target column 'label'
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != "label"]

    if not feature_cols:
        raise RuntimeError(
            "No numeric feature columns found in dataset. "
            "Check that build_features.py is producing numeric features."
        )

    return feature_cols


# ============================================================================
# TRAIN/TEST SPLIT
# ============================================================================

def load_and_split(
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the cleaned dataset and return an 80/20 stratified train/test split.

    Returns:
        X_train, X_test, y_train, y_test
    """
    df = load_features()
    feature_cols = infer_feature_columns(df)

    X = df[feature_cols].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print("Loaded cleaned dataset from:", CLEANED_FEATURES_PATH)
    print(f"  Total rows: {df.shape[0]}")
    print(
        f"  Feature columns ({len(feature_cols)}): "
        f"{feature_cols[:10]}{' ...' if len(feature_cols) > 10 else ''}"
    )
    print(
        f"Train/test split done. "
        f"Train size = {X_train.shape[0]}, Test size = {X_test.shape[0]}"
    )

    return X_train, X_test, y_train, y_test


# ============================================================================
# GLOBAL FEATURE_COLS (for main.py)
# ============================================================================

try:
    _df_tmp = load_features()
    FEATURE_COLS: List[str] = infer_feature_columns(_df_tmp)
except FileNotFoundError:
    # If build_features hasn't been run yet, keep this empty;
    # main.py will fail later with a clear error from load_features().
    FEATURE_COLS = []


# ============================================================================
# DEBUG / CLI
# ============================================================================

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_split()
    print("X_train shape:", X_train.shape)
    print("X_test  shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test  shape:", y_test.shape)
    print("FEATURE_COLS length:", len(FEATURE_COLS))