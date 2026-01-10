# Fantasy Football Stock Market Simulator
**Machine Learning–Based Match Outcome Prediction and Stock-Style Team Valuation in the Premier League**  
**Data Science & Advanced Programming – Model Comparison & Forecasting**

---

## Research Question

This project investigates whether historical football match data can be used to (i) predict Premier League match outcomes using supervised machine-learning classification models, and (ii) translate on-field performance and short-term expectations into a stock-market-style pricing mechanism for football clubs.

Rather than focusing purely on predictive accuracy, the project emphasises interpretability and realism, exploring how probabilistic match forecasts can be transformed into intuitive “buy”, “hold”, and “sell”-style signals and visual price trajectories that resemble financial assets.

---

## Project Overview

The project implements a complete end-to-end data science pipeline, starting from raw historical Premier League match data and ending with interactive visualisations and stock-style price outputs. Match-level data is first cleaned and transformed into team-level observations through feature engineering. These features are then used to train and evaluate multiple supervised classification models that predict match outcomes (win, draw, loss).

Four models are compared: Logistic Regression, k-Nearest Neighbours (KNN), Random Forest, and Gradient Boosting. Their predictive performance is evaluated using accuracy, F1-scores, and confusion matrices. The best-performing model is then used to generate probabilistic forecasts for future fixtures in the 2025–2026 season.

These probabilistic predictions are subsequently mapped into expected points, directional stock signals, and a pricing engine that updates each team’s “stock price” over time. The final outputs include both static summary figures and interactive Plotly visualisations that allow users to explore historical price paths and upcoming match predictions.

All stages of the pipeline are orchestrated through a single entry point (`main.py`) to ensure reproducibility and ease of execution.

---

## Repository Structure

```text
fantasy-football-stock-market-simulator/
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── evaluation.py
│   ├── models.py
│   ├── predict_future.py
│   ├── stock_direction.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── build_features.py
│   ├── price_engine/
│   │   ├── __init__.py
│   │   └── pricing_engine.py
│   ├── graphs/
│   │   ├── __init__.py
│   │   ├── plot_stock_chart.py
│   │   └── plot_stock_signals.py
│   └── utils/
│       ├── __init__.py
│       └── team_names.py
├── data/
│   ├── raw/
│   └── cleaned_data/
├── docs/
│   ├── fantasy_football_stock_market_simulator_report.pdf
│   ├── fantasy_football_stock_market_simulator_report.tex
│   └── report_figures/
├── results/
│   ├── model_eval/
│   ├── forecasts/
│   ├── pricing_engine/
│   └── charts/
├── main.py
├── PROPOSAL.md
├── README.md
├── requirements.txt
└── environment.yml
```

---

## Project Report

A full written report accompanying this project is included in the repository.
	•	Precompiled PDF (recommended for viewing):
docs/fantasy_football_stock_market_simulator.pdf
	•	LaTeX source files:
docs/fantasy_football_stock_market_simulator.tex
docs/references.bib

The PDF contains the complete academic write-up of the project. The LaTeX source is provided for transparency and reproducibility but does not need to be compiled in order to run the code.

Note: The report is independent of the Python pipeline. It is not executed by main.py and does not require LaTeX to be installed. The PDF is included solely for reading and assessment purposes after cloning the repository.

---

## Environment Setup

The project can be run using either a standard Python virtual environment or a Conda environment. Both approaches install the same dependencies and produce identical results.

For a local Python virtual environment, create and activate the environment and install dependencies as follows:
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate         # Windows
pip install -r requirements.txt

Alternatively, a Conda environment can be created using:
conda env create -f environment.yml
conda activate football-stock-project

---

## Running the Project

From the project root directory, the entire pipeline can be executed with a single command:
This command rebuilds features, trains and evaluates all models, selects the best-performing classifier, generates future match predictions, computes stock-direction signals, runs the pricing engine, and saves all outputs to the `results/` directory.

---

## Outputs

The project produces several categories of outputs. Model evaluation results include accuracy summaries, classification reports, and confusion matrices for each classifier. Forecast outputs include probabilistic predictions for upcoming 2025–2026 fixtures and derived stock-direction signals. The pricing engine generates historical and current stock-style price series for each team.

Visual outputs include static charts such as correlation heatmaps and signal rankings, as well as an interactive Plotly chart (`next_week_stock_overlay.html`) that overlays historical prices with upcoming match predictions and probabilities The interactive visualisations are generated as standalone HTML files using Plotly.

Note on Nuvolos / remote environments:
Because Nuvolos does not support opening a web browser automatically, interactive charts are saved to disk rather than opened directly.

After running `main.py`, open the interactive dashboard manually:
results/charts/next_week_stock_overlay.html
Download this file and open it locally in any modern web browser (Chrome, Safari, Firefox).

---

## Notes on Reproducibility

Randomness is controlled where applicable to ensure consistent results across runs. All results can be regenerated at any time by re-running `main.py`, and no manual intervention is required once the environment is set up. The `results/` directory is treated as generated output and does not need to be version-controlled. Team names are normalised across all modules to ensure consistency between datasets and visualisations.

---

## Key Technologies

The project is implemented in Python 3.11 and makes extensive use of the pandas and numpy libraries for data manipulation. Machine-learning models are built using scikit-learn, while visualisations are created with matplotlib, seaborn, and Plotly (for interactive charts).

---

## Author

**Saarujan Sivananth**  
MSc Finance  
Data Science & Advanced Programming
