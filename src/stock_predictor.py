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

# target variable:
# if tomorrow's price > today's price, 1
# else, 0
data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
print("\nWith target column:")
print(data[["Close", "Target"]].head())

# drop rows with NaN from rolling calculations
data = data.dropna()

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