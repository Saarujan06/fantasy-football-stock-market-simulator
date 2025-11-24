PROPOSAL
Fantasy Football Stock Market Simulator
Category: Data Science, Predictive Modelling, Sports Analytics

⸻

1. Problem Statement / Motivation

This project aims to transform Premier League match performance into a financial-style “stock price” that reflects each club’s perceived strength, momentum, and expected future performance. Football fans often speak about clubs in financial metaphors—“their stock is rising”, “their value has collapsed”, “momentum is strong”—but there is no quantitative measure that behaves like a real asset price.

By creating a football stock market, this project provides an interpretable, data-driven metric of team performance over time. It combines statistical modelling with a real-world sports context, providing an engaging way to explore regression, feature engineering, and predictive analysis using Python.

⸻

2. Planned Approach and Technologies

The project is divided into five stages.
First, raw match data from the 2020–2026 Premier League seasons is cleaned and merged into a single dataset with standardised column names and dates. This produces a unified foundation of match-level information (goals, discipline, shots, etc.).

Second, feature engineering converts match records into team-level observations. Each row becomes a team-match entry including points earned, goal difference, clean sheets, disciplinary points, rolling form, and opponent strength. This produces a structured panel dataset suitable for modelling.

Third, each team is assigned a starting price for the 2025–2026 season. Instead of giving all teams identical initial values, the starting price equals the final price from the previous season. This captures long-term expectations and club identity—Manchester City begin high, struggling or newly promoted clubs begin lower.

Fourth, an OLS regression is fitted using all seasons other than 2025–2026. The model predicts expected points from features such as xGD, clean sheets, opponent strength, and form. During the 2025–2026 season, each match updates a team’s price based on both the actual result and the model’s predicted performance. The price change formula combines realised performance with prediction error, producing realistic, stable fluctuations throughout the season.

Finally, visualisation tools allow the user to request a team and instantly display its stock-price graph, full historical trajectory, and current ranking. Additional scripts plot all teams together and create sortable tables of end-of-season valuations.

The project is implemented entirely in Python using pandas, numpy, and matplotlib, following a clear modular folder structure.

⸻

3. Expected Challenges and Mitigation

Football data varies across seasons, particularly in the presence or absence of xG features. This is controlled by strict validation during cleaning and fallback rules for missing values. Pricing models can become unstable if deviations accumulate, so scaling and mean-centering ensure volatility remains realistic. Predictive leakage is avoided by training the regression only on seasons prior to the one being priced.

⸻

4. Success Criteria

The system should generate intuitive stock-price curves that rise in strong periods, fall in weak periods, and reflect each team’s historical identity. The regression should meaningfully influence price evolution, and visual outputs must be clear and reproducible. The full pipeline—from raw data to graphs—should run consistently with minimal manual intervention.

⸻

5. Stretch Goals

If time allows, the model can be extended to next-match probabilistic forecasting, portfolio simulations, or a small web dashboard for interactive exploration.