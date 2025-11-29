# src/models/match_prediction_model.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


@dataclass
class MatchPredictionMetrics:
    """Container for model performance metrics."""
    r2: float
    r: float        # Pearson correlation coefficient
    n_train: int
    n_val: int


class MatchPredictionModel:
    """
    Model 2: Machine-learning regression model that predicts expected points
    (0, 1, or 3) for each team–match observation.

    This is trained on historical `team_match_features.csv` and then used to
    estimate expected points for future fixtures.

    Targets:
      y_t = pts_t  (actual points from the match)

    Features (you can extend this list later):
      - xGD              (xg_for - xg_against)
      - form3            (rolling avg points last 3 matches)
      - opp_avg_pts      (rolling avg opponent strength)
      - card_points      (discipline)
      - is_home          (1 = home, 0 = away)
    """

    def __init__(self, random_state: int = 42):
        # RandomForest is flexible and handles nonlinearities
        self.model = RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )
        self.feature_cols: List[str] = [
            "xGD",
            "form3",
            "opp_avg_pts",
            "card_points",
            "is_home",
        ]
        self.metrics: MatchPredictionMetrics | None = None
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, df: pd.DataFrame, test_size: float = 0.2) -> MatchPredictionMetrics:
        """
        Fit the model on historical matches.

        Args:
            df: team-match feature DataFrame. Must contain columns:
                - 'pts' (target)
                - feature columns in self.feature_cols
            test_size: fraction of data held out as validation.

        Returns:
            MatchPredictionMetrics with R² and Pearson r on validation data.
        """
        missing = [c for c in self.feature_cols + ["pts"] if c not in df.columns]
        if missing:
            raise ValueError(f"Training data missing required columns: {missing}")

        # Drop rows with NaNs in features/target
        df_train = df.dropna(subset=self.feature_cols + ["pts"]).copy()
        if df_train.empty:
            raise ValueError("No training rows after dropping NaNs.")

        X = df_train[self.feature_cols].to_numpy(dtype=float)
        y = df_train["pts"].to_numpy(dtype=float)

        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=test_size, shuffle=True, random_state=42
        )

        self.model.fit(X_tr, y_tr)
        self._is_fitted = True

        # Validation predictions
        y_pred = self.model.predict(X_val)

        r2 = float(r2_score(y_val, y_pred))
        # Pearson correlation coefficient r
        if len(y_val) > 1:
            r = float(np.corrcoef(y_val, y_pred)[0, 1])
        else:
            r = float("nan")

        self.metrics = MatchPredictionMetrics(
            r2=r,
            r=r,
            n_train=len(y_tr),
            n_val=len(y_val),
        )

        print("\n=== MatchPredictionModel: validation performance ===")
        print(f"  R²   : {r2:.4f}")
        print(f"  r    : {r:.4f}")
        print(f"  Ntrain = {len(y_tr)}, Nval = {len(y_val)}\n")

        return self.metrics

    # ------------------------------------------------------------------
    # Prediction for new matches
    # ------------------------------------------------------------------
    def predict_expected_points(self, df_features: pd.DataFrame) -> np.ndarray:
        """
        Predict expected points for new team–match rows (e.g., future fixtures).

        Args:
            df_features: DataFrame with the same feature columns as training:
                         ['xGD', 'form3', 'opp_avg_pts', 'card_points', 'is_home']

        Returns:
            np.ndarray of shape (n_rows,) with predicted expected points.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be trained before calling predict_expected_points().")

        for col in self.feature_cols:
            if col not in df_features.columns:
                raise ValueError(f"Missing feature column in prediction data: {col}")

        X_new = df_features[self.feature_cols].to_numpy(dtype=float)
        return self.model.predict(X_new)

    # ------------------------------------------------------------------
    # Convenience: fit + predict in one go
    # ------------------------------------------------------------------
    def fit_and_predict(
        self,
        df_train: pd.DataFrame,
        df_future: pd.DataFrame,
        test_size: float = 0.2,
    ) -> Tuple[np.ndarray, MatchPredictionMetrics]:
        """
        Train the model on df_train and then predict expected points for
        df_future in one step.

        Returns:
            (predicted_points_for_future, metrics_on_validation)
        """
        metrics = self.train(df_train, test_size=test_size)
        preds_future = self.predict_expected_points(df_future)
        return preds_future, metrics