# Fantasy Football “Stock Market” Simulator

## **Category**
Data Analysis & Visualization • Simulation & Modeling • Predictive Analytics  

---

## **1. Project Overview**

This project models English Premier League (EPL) teams as if they were stocks whose prices change weekly based on performance. Each team’s “stock price” will fluctuate according to match results, expected goals (xG), and opponent strength, mimicking how financial markets respond to company fundamentals.  

The objective is to create a system that translates football performance data into measurable financial-style indicators such as returns, volatility, and momentum. Users can visualize how team values evolve throughout a season and analyze which performance factors drive price movements.  

To meet the data-science requirement, a **predictive component** will also be added. Using regression or time-series models (e.g., linear regression or ARIMA), the system will forecast each team’s next-week “price change” from recent features such as results, xG, xGA, and opponent strength. This forward-looking layer enables simulated “investment” decisions based on predicted performance rather than only historical outcomes.  

---

## **2. Data Sources**

The analysis focuses exclusively on the English Premier League to maintain consistent structure and comparability across teams. All data will come from open, machine-readable public sources verified to cover 2019–2024:  

| Source | Coverage | Key Variables | Link |
|--------|-----------|---------------|------|
| **Football-Data.co.uk / EPL (E0)** | 2000 – 2025 | Match date, teams, goals, shots, corners, fouls, yellow/red cards, betting odds | [football-data.co.uk](https://www.football-data.co.uk/data.php) |
| **Understat / Kaggle Team & Player Metrics** | 2015 – 2024 | Team & player xG, xGA, PPDA, deep completions | [Kaggle dataset](https://www.kaggle.com/datasets/codytipton/player-stats-per-game-understat) |
| **FiveThirtyEight SPI/xG Dataset** | 2016 – present | xG, SPI ratings, win/draw probabilities | [538 GitHub](https://github.com/fivethirtyeight/data/tree/master/soccer-spi) |

A Python script will automatically verify and download season files from Football-Data and SPI for 2019–2024, merge them into a single dataset, and align team names and dates.  

---

## **3. Planned Approach**

1. **Data Processing:** download, clean, and merge Football-Data results with SPI/Understat xG metrics.  
2. **Price Modeling:** derive weekly “stock prices” from match outcomes and performance indicators.  
3. **Visualization:** plot team price trajectories, volatility, and form-adjusted indices with *matplotlib/seaborn*.  
4. **Prediction:** train regression or ARIMA models to forecast next-week price changes; evaluate using MAE and RMSE.  
5. **Evaluation:** compare predicted vs. actual price movement and analyze key feature importance.  

---

## **4. Expected Challenges**

- Aligning team names across sources (Football-Data, SPI, Understat).  
- Missing xG data for some matches.  
- Avoiding model overfitting with limited seasons.  
- Designing an intuitive price-update formula.  

---

## **5. Success Criteria**

- Working Python pipeline producing a clean merged dataset (2019–2024).  
- Accurate weekly price calculation & visualization for all EPL teams.  
- Predictive model generating reasonable next-week forecasts.  
- Well-structured, documented, and tested codebase (> 70 % coverage).  
 