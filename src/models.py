from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =============================================================================
# 1. LOGISTIC REGRESSION (WITH STANDARDISATION)
# =============================================================================


def train_logistic_regression(
    X_train: np.ndarray, y_train: np.ndarray
) -> Tuple[Any, Dict[str, Any]]:
    lr_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=1000,
                ),
            ),
        ]
    )

    param_grid = {
        "clf__C": [0.1, 0.5, 1.0],
    }

    grid = GridSearchCV(
        estimator=lr_pipeline,
        param_grid=param_grid,
        scoring="balanced_accuracy",
        cv=5,
        n_jobs=-1,
    )

    print("\nTraining Logistic Regression...")
    grid.fit(X_train, y_train)

    print("Best params for Logistic Regression:", grid.best_params_)

    best_model = grid.best_estimator_
    best_params = grid.best_params_

    return best_model, best_params


# =============================================================================
# 2. RANDOM FOREST
# =============================================================================


def train_random_forest(
    X_train: np.ndarray, y_train: np.ndarray
) -> Tuple[Any, Dict[str, Any]]:
    rf = RandomForestClassifier(random_state=0)

    param_grid = {
        "n_estimators": [100, 300],
        "max_depth": [None, 10],
        "min_samples_leaf": [1, 2],
    }

    grid = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring="balanced_accuracy",
        cv=5,
        n_jobs=-1,
    )

    print("\nTraining Random Forest...")
    grid.fit(X_train, y_train)

    print("Best params for Random Forest:", grid.best_params_)

    best_model = grid.best_estimator_
    best_params = grid.best_params_

    return best_model, best_params


# =============================================================================
# 3. K-NEAREST NEIGHBOURS (WITH STANDARDISATION)
# =============================================================================


def train_knn(
    X_train: np.ndarray, y_train: np.ndarray
) -> Tuple[Any, Dict[str, Any]]:
    knn_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier()),
        ]
    )

    param_grid = {
        "clf__n_neighbors": [5, 9, 15],
        "clf__weights": ["uniform", "distance"],
    }

    grid = GridSearchCV(
        estimator=knn_pipeline,
        param_grid=param_grid,
        scoring="balanced_accuracy",
        cv=5,
        n_jobs=-1,
    )

    print("\nTraining KNN...")
    grid.fit(X_train, y_train)

    print("Best params for KNN:", grid.best_params_)

    best_model = grid.best_estimator_
    best_params = grid.best_params_

    return best_model, best_params


# =============================================================================
# 4. GRADIENT BOOSTING (NEW MODEL)
# =============================================================================


def train_gradient_boosting(
    X_train: np.ndarray, y_train: np.ndarray
) -> Tuple[Any, Dict[str, Any]]:
    gb = GradientBoostingClassifier(random_state=0)

    param_grid = {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [2, 3],
    }

    grid = GridSearchCV(
        estimator=gb,
        param_grid=param_grid,
        scoring="balanced_accuracy",
        cv=5,
        n_jobs=-1,
    )

    print("\nTraining Gradient Boosting...")
    grid.fit(X_train, y_train)

    print("Best params for Gradient Boosting:", grid.best_params_)

    best_model = grid.best_estimator_
    best_params = grid.best_params_

    return best_model, best_params