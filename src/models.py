"""
Model training utilities for the fantasy football stock market project.

Provides separate training functions for:
    - Logistic Regression  (with StandardScaler + GridSearchCV)
    - Random Forest        (GridSearchCV)
    - K-Nearest Neighbours (with StandardScaler + GridSearchCV)
    - Gradient Boosting    (GridSearchCV)

Each function returns:
    best_model, best_params
where best_model is the fitted estimator (ready for predict / predict_proba)
and best_params is the dict of best hyperparameters.
"""

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
    """
    Train a multinomial Logistic Regression model on the training data.

    We use:
        - StandardScaler (features standardised to mean 0, std 1)
        - LogisticRegression(multi_class='multinomial', solver='lbfgs')
        - max_iter=1000 to avoid convergence issues
        - Grid search over C

    Returns
    -------
    best_model : sklearn Pipeline
        Pipeline(scaler, logistic regression) fitted on the full training set.
    best_params : dict
        Best hyperparameter combination found by GridSearchCV.
    """

    lr_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    multi_class="multinomial",
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
        scoring="accuracy",
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
    """
    Train a RandomForestClassifier with a small hyperparameter grid.

    We do NOT scale features here because tree-based models are
    invariant to monotonic transformations of individual features.

    Returns
    -------
    best_model : RandomForestClassifier
    best_params : dict
    """

    rf = RandomForestClassifier(random_state=0)

    param_grid = {
        "n_estimators": [100, 300],
        "max_depth": [None, 10],
        "min_samples_leaf": [1, 2],
    }

    grid = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring="accuracy",
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
    """
    Train a K-Nearest Neighbours classifier.

    We wrap KNN in a Pipeline with StandardScaler because KNN is
    distance-based and very sensitive to feature scales.

    Returns
    -------
    best_model : sklearn Pipeline
        Pipeline(scaler, KNeighborsClassifier).
    best_params : dict
    """

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
        scoring="accuracy",
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
    """
    Train a GradientBoostingClassifier as an additional model.

    Gradient boosting is often strong on tabular data and can
    sometimes outperform Random Forests.

    We use a small parameter grid to keep training time reasonable.

    Returns
    -------
    best_model : GradientBoostingClassifier
    best_params : dict
    """

    gb = GradientBoostingClassifier(random_state=0)

    param_grid = {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [2, 3],
    }

    grid = GridSearchCV(
        estimator=gb,
        param_grid=param_grid,
        scoring="accuracy",
        cv=5,
        n_jobs=-1,
    )

    print("\nTraining Gradient Boosting...")
    grid.fit(X_train, y_train)

    print("Best params for Gradient Boosting:", grid.best_params_)

    best_model = grid.best_estimator_
    best_params = grid.best_params_

    return best_model, best_params