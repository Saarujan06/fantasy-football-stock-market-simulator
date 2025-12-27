# Project Proposal  
**Fantasy Football Stock Market Simulator**  
Category: Data Science, Machine Learning, Sports Analytics

---

## 1. Problem Statement and Motivation

Football teams are often discussed using financial metaphors: clubs are said to be “in form”, “losing value”, or “on the rise”. While such language is common among fans and analysts, there is no widely used quantitative framework that translates football performance into a financial-style signal that evolves over time and explicitly incorporates uncertainty.

The objective of this project is to construct a fantasy football stock market by applying supervised machine-learning techniques to predict football match outcomes and transform those probabilistic predictions into short-term stock-style performance signals. Rather than focusing solely on prediction accuracy, the project aims to explore how uncertainty-aware model outputs can be repurposed into interpretable decision signals, linking sports analytics with concepts commonly used in quantitative finance.

This approach allows the project to address both a methodological question—how well machine-learning models can predict football match outcomes—and a practical one—how such predictions can be translated into meaningful indicators of future team performance.

---

## 2. Data and Feature Construction

The analysis is based on historical Premier League data spanning from the 2010–2011 season to the most recent completed fixtures available at runtime. Match results, fixtures, and betting odds are obtained from publicly available football data sources, while expected goals (xG) and advanced performance statistics are incorporated where available. Since xG data only becomes consistently available from the 2014–2015 season onwards, earlier seasons are handled using conservative assumptions to preserve temporal consistency.

Raw data are cleaned and merged into two structured datasets: a season-level dataset and a match-level dataset organised from a team-centric perspective. Each match is represented twice, once for each team involved, allowing the prediction problem to be framed at the team–match level. Particular care is taken to ensure that all features represent information available prior to kickoff.

Feature engineering focuses on rolling performance metrics that capture recent form, including expected goals, goals conceded, points earned, and pressing indicators. Betting odds are transformed into implied probabilities and included as features to reflect market expectations. All rolling statistics are computed using lagged values to prevent information leakage, ensuring that models only have access to past information when making predictions.

---

## 3. Modelling Approach

The predictive task is formulated as a multi-class classification problem, where the target variable represents match outcomes from a team’s perspective: loss, draw, or win. This framing aligns naturally with the discrete nature of football results and allows probabilistic predictions to be generated for each outcome.

Four supervised machine-learning classifiers are evaluated: Logistic Regression, k-Nearest Neighbours (KNN), Random Forest, and Gradient Boosting. These models were selected to provide a balance between linear and non-linear approaches, as well as between interpretable baselines and more flexible ensemble methods.

All models are trained using a chronological train–test split to preserve the temporal structure of the data and avoid look-ahead bias. Identical feature sets and preprocessing steps are applied across models to ensure comparability. Hyperparameters are selected using cross-validation on the training set, and model performance is evaluated on a held-out test set using accuracy, balanced accuracy, and F1-scores, with both macro-averaged and weighted metrics reported to account for class imbalance.

---

## 4. From Match Predictions to Stock-Style Signals

Rather than relying solely on hard class predictions, the probabilistic outputs of the classifiers play a central role in the project. Predicted probabilities for wins, draws, and losses are transformed into expected points and uncertainty measures, which are then aggregated at the team level to produce short-term stock-style performance signals.

These signals are visualised using interactive charts that overlay historical team “price” trajectories with predicted outcomes for the upcoming matchweek. The colour and annotations of each trajectory reflect the model’s next-match prediction and associated uncertainty, providing an intuitive link between machine-learning outputs and financial-style forecasting concepts.

This transformation demonstrates how classification probabilities can be repurposed into interpretable decision-support tools, extending the usefulness of predictive models beyond simple outcome prediction.

---

## 5. Success Criteria and Scope

The project is considered successful if the evaluated machine-learning models outperform a random baseline in predicting match outcomes, with ensemble methods demonstrating improved performance over simpler classifiers. While draw outcomes are expected to remain challenging to predict, their probabilistic representation should meaningfully reflect uncertainty rather than being ignored entirely.

In addition, the generated stock-style signals should be intuitive, stable, and reproducible, and the full pipeline—from raw data to evaluation reports and interactive visualisations—should execute consistently with minimal manual intervention.

---

## 6. Future Extensions

Possible extensions include probability calibration to improve draw prediction, explicit price-update rules driven by expected points, portfolio-style simulations across teams, and deployment as a lightweight interactive dashboard. These extensions build naturally on the probabilistic framework developed in this project and offer clear directions for further research.