# Stock Direction Model

## Overview

This project builds a machine learning pipeline to predict the **next-day price direction** (up/down) of a stock using historical price data and technical indicators.

A relatively volatile asset, Palantir (PLTR), is used as the primary test case to evaluate how well the model performs under higher-noise conditions.

The goal is not to predict exact prices, but to classify short-term movement and evaluate whether these predictions can be translated into a **profitable trading strategy** through backtesting.

---

## Dataset

- Ticker: PLTR  
- Start Date: 2020-01-01  
- End Date: 2025-12-31  
- Total Data: ~6 years of daily price data (~1300 observations)

Data is sourced using yfinance and includes:

- Open, High, Low, Close, Volume  

---

## Feature Engineering

Raw price data is transformed into technical indicators to capture different market signals:

- Return: daily percentage price change (momentum)
- MA_5 / MA_20: short-term and medium-term moving averages (trend)
- Volatility: rolling standard deviation of returns (risk)
- Volume_Change: change in trading volume (activity)
- MA_Ratio: ratio of short-term to long-term trend strength
- Momentum_3: 3-day price momentum

To prevent data leakage, all features are **shifted by one day**, ensuring the model only uses past information when making predictions.

---

## Target Variable

The model predicts:

```
1 → next day’s close > today’s close  
0 → otherwise
```

This frames the problem as a **binary classification task**.

---

## Model

A Random Forest Classifier is used as a baseline model due to its strong performance on tabular data and minimal tuning requirements.

- n_estimators = 100  
- time-based train/test split (80/20)

---

## Results

### Model Performance

- Baseline model accuracy: ~0.49  
- With additional features: ~0.50  

While the increase in accuracy is marginal, this is expected in short-term stock prediction due to high noise and low signal.

Additionally, the relatively low accuracy is consistent with the high noise and volatility present in short-term stock movements, where price changes are often driven by unpredictable factors, resulting in a low signal-to-noise ratio.

However, additional features improved **recall for upward movements**, indicating better identification of positive price changes.

---

## Feature Importance

Feature importance analysis shows that:

- Momentum_3 (short-term momentum) is the most influential feature  
- Followed by:
  - Volume_Change  
  - MA_Ratio  
  - Volatility  

This suggests that recent price momentum and trading activity are more informative than raw returns.

---

## Backtesting

A simple strategy is evaluated:

```
If model predicts UP → invest  
Else → stay out
```

### Results

- Strategy return: ~2.7–3.0x  
- Market return: ~2.4–2.5x  

Despite near-random classification accuracy, the model outperforms a buy-and-hold strategy over the test period.

This highlights an important insight:

Even weak predictive models can be useful if they capture higher-impact movements or avoid downside risk.

---

## Market Insight

One notable trend in the dataset is the strong upward movement in PLTR during the 2024–2025 period.

This aligns with increased interest in:
- large-scale data platforms  
- enterprise analytics  
- AI-related infrastructure  

During sustained upward trends, momentum-based features (such as Momentum_3 and MA_Ratio) become more predictive, which is consistent with the model assigning them higher importance.

---

## Limitations

- Short-term stock movements are highly stochastic  
- Model performance is close to random baseline  
- Results are based on a single asset (PLTR)  
- No transaction costs or slippage included  
- No macroeconomic or fundamental data used  

---

## How to Run

```
pip install -r requirements.txt
python src/stock_predictor.py
```

