*Fantasy Football Stock Market Simulator
Data Science & Advanced Programming – Model Comparison & Forecasting

Research Question

Can football match data be used to:
	1.	Predict match outcomes using supervised classification models, and
	2.	Translate on-field performance into a stock-market-style pricing mechanism for Premier League teams?

We compare multiple machine-learning classifiers and use their predictions to generate future forecasts, stock signals, and price trajectories.

⸻

Project Overview

This project implements a full end-to-end data science pipeline:
	1.	Feature engineering from historical Premier League match data
	2.	Model training & evaluation
	    •	Logistic Regression
	    •	K-Nearest Neighbors
	    •	Random Forest
	    •	Gradient Boosting
	3.	Model comparison using accuracy and confusion matrices
	4.	Future match prediction (2025–26 fixtures)
	5.	Stock signal generation (BUY / HOLD / SELL)
	6.	Football stock pricing engine
	7.	Static and interactive visualisations

All steps are orchestrated via a single entry point: main.py.

⸻

Repository Structure
src/
  pipeline/
    build_features.py
  price_engine/
    pricing_engine.py
  models/
  graphs/
  utils/

data/
  raw/
  cleaned_data/

results/
  model_eval/
  forecasts/
  pricing_engine/
  charts/

main.py
requirements.txt
environment.yml
README.md

⸻

Environment Setup

Option A — Local Python (venv) ✅ (recommended for local users)
python -m venv .venv
source .venv/bin/activate        # Mac / Linux
.venv\Scripts\activate         # Windows

pip install -r requirements.txt

Option B — Conda (Nuvolos)
conda env create -f environment.yml
conda activate football-stock-project

Both environments install the same dependencies.

Running the Project

From the project root:
python main.py

This single command will:
	1.	Rebuild match-level features (latest season included)
	2.	Train and evaluate all models
	3.	Select the best-performing model
	4.	Predict the next matchweek
	5.	Generate stock direction signals
	6.	Run the pricing engine
	7.	Save results and charts to the results/ directory

⸻

Outputs

Model Evaluation
	•	Accuracy CSVs
	•	Classification reports
	•	Confusion matrices

Forecasts
	•	future_predictions_2025_26.csv
	•	stock_direction_2025_26.csv

Pricing Engine
	•	price_timeseries.csv
	•	current_prices_2025_2026.csv

Visualisations
	•	Feature correlation heatmaps
	•	Future probability bar charts & tables
	•	Stock signal bar charts
	•	Interactive Plotly stock chart (next_week_stock_overlay.html)

⸻

Notes for Reproducibility
	•	All randomness is controlled where applicable.
	•	Results can be regenerated at any time by re-running main.py.
	•	The results/ directory does not need to be version-controlled.
	•	Team names are normalized across all modules to ensure consistency.

⸻

Key Technologies
	•	Python 3.11
	•	pandas, numpy
	•	scikit-learn
	•	matplotlib, seaborn
	•	plotly (optional, for interactive charts)

⸻

Author

Saarujan Sivananth
MSc Finance / Data Science & Advanced Programming