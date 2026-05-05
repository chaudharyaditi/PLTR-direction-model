import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# download historical stock data for PLTR
ticker = "PLTR"
data = yf.download(
    ticker,
    start="2020-01-01",
    end="2026-01-01"
)
data.columns = data.columns.get_level_values(0) # flatten the column names if they are multi-level

# daily return (momentum)
data["Return"] = data["Close"].pct_change()

# 5-day moving average (trend indicator)
data["MA_5"] = data["Close"].rolling(5).mean()

# 20-day moving average (trend indicator)
data["MA_20"] = data["Close"].rolling(20).mean()

# volatility (5-day std of returns) (risk indicator)
data["Volatility"] = data["Return"].rolling(5).std()

# volume change (momentum in trading activity)
data["Volume_Change"] = data["Volume"].pct_change()

# moving average ratio (trend strength)
data["MA_Ratio"] = data["MA_5"] / data["MA_20"]

# price momentum over 3 days
data["Momentum_3"] = data["Close"] / data["Close"].shift(3)

# target variable:
# if tomorrow's price > today's price, 1
# else, 0
data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
print("\nWith target column:")
print(data[["Close", "Target"]].head())

# drop rows with NaN from rolling calculations
data = data.dropna()

# features to use
features = ["Return", "MA_5", "MA_20", "Volatility", "Volume_Change", "MA_Ratio", "Momentum_3"]

# we have to shift features so we only use past data
data[features] = data[features].shift(1)
data = data.dropna()

X = data[features]
y = data["Target"]

# split data into training and testing sets (80% train, 20% test)
split_index = int(len(data) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# train
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# evaluate the model
from sklearn.metrics import accuracy_score, classification_report

predictions = model.predict(X_test)

print("\nModel Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# feature importance
import matplotlib.pyplot as plt

importances = model.feature_importances_

plt.figure(figsize=(8, 5))
plt.barh(features, importances)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.show()

print("\nAfter feature engineering:")
print(data.head())

print("First 5 rows:")
print(data.head()) # display the first 5 rows of the dataset to get overview of the data

print("\nLast 5 rows:")
print(data.tail()) # display the last 5 rows of the dataset to check for any recent trends or anomalies

print("\nDataset shape:")
print(data.shape) # display the shape of the dataset to understand how many rows and columns it contains

print("\nMissing values:")
print(data.isnull().sum()) # display the number of missing values in each column

print("\nColumns:")
print(data.columns) # display the column names to understand what data is available

# plot the closing price over time
plt.figure(figsize=(10, 5))
plt.plot(data.index, data["Close"])
plt.title("PLTR Closing Price Over Time")
plt.xlabel("Date")
plt.ylabel("Closing Price")
plt.grid(True)
plt.show()

# backtesting
# simulating: if model predicts UP, buy
# else, do nothing
test_data = data.iloc[split_index:].copy()
test_data["Prediction"] = predictions

# strategy returns:
# if prediction = 1 → take next day's return
# else → 0
test_data["Strategy_Return"] = test_data["Return"] * test_data["Prediction"]

# cumulative returns
test_data["Cumulative_Strategy"] = (1 + test_data["Strategy_Return"]).cumprod()
test_data["Cumulative_Market"] = (1 + test_data["Return"]).cumprod()

# plot
plt.figure(figsize=(10, 5))
plt.plot(test_data.index, test_data["Cumulative_Strategy"], label="Strategy")
plt.plot(test_data.index, test_data["Cumulative_Market"], label="Market")
plt.legend()
plt.title("Strategy vs Market Performance")
plt.show()