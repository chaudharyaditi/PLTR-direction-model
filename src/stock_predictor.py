import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# config
TICKER = "PLTR"
START_DATE = "2020-01-01"
END_DATE = "2026-01-01"
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 100

FEATURES = ["Return", "MA_5", "MA_20", "Volatility", "Volume_Change", "MA_Ratio", "Momentum_3"]


def load_data():
    # download historical stock data for PLTR
    data = yf.download(
        TICKER,
        start=START_DATE,
        end=END_DATE
    )
    data.columns = data.columns.get_level_values(0) # flatten the column names if they are multi-level

    return data


def create_features(data):
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

    return data


def create_target(data):
    # target variable:
    # if tomorrow's price > today's price, 1
    # else, 0
    data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
    print("\nWith target column:")
    print(data[["Close", "Target"]].head())

    return data


def prepare_data(data):
    # drop rows with NaN from rolling calculations
    data = data.dropna()

    # features to use
    features = FEATURES

    # we have to shift features so we only use past data
    data[features] = data[features].shift(1)
    data = data.dropna()

    X = data[features]
    y = data["Target"]

    return data, X, y, features


def split_data(data, X, y):
    # split data into training and testing sets (80% train, 20% test)
    split_index = int(len(data) * (1 - TEST_SIZE))

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    return split_index, X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    # train
    model = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):
    # evaluate the model
    predictions = model.predict(X_test)

    print("\nModel Accuracy:", accuracy_score(y_test, predictions))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    # save model results
    with open("results/model_results.txt", "w") as f:
        f.write(f"Ticker: {TICKER}\n")
        f.write(f"Model Accuracy: {accuracy_score(y_test, predictions)}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, predictions))

    return predictions


def plot_feature_importance(model, features):
    # feature importance
    importances = model.feature_importances_

    plt.figure(figsize=(8, 5))
    plt.barh(features, importances)
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.savefig("images/feature_importance.png")  # SAVE
    plt.show()


def inspect_data(data):
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


def plot_closing_price(data):
    # plot the closing price over time
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data["Close"])
    plt.title("PLTR Closing Price Over Time")
    plt.xlabel("Date")
    plt.ylabel("Closing Price")
    plt.grid(True)
    plt.savefig("images/closing_price.png")  # SAVE
    plt.show()


def backtest_strategy(data, split_index, predictions):
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
    plt.savefig("images/strategy_vs_market.png")  # SAVE
    plt.show()

    strategy_return = test_data["Cumulative_Strategy"].iloc[-1]
    market_return = test_data["Cumulative_Market"].iloc[-1]

    with open("results/backtest_results.txt", "w") as f:
        f.write(f"Strategy Final Return: {strategy_return}\n")
        f.write(f"Market Final Return: {market_return}\n")


def main():
    data = load_data()
    data = create_features(data)
    data = create_target(data)
    data, X, y, features = prepare_data(data)

    split_index, X_train, X_test, y_train, y_test = split_data(data, X, y)

    model = train_model(X_train, y_train)
    predictions = evaluate_model(model, X_test, y_test)

    plot_feature_importance(model, features)
    inspect_data(data)
    plot_closing_price(data)
    backtest_strategy(data, split_index, predictions)


if __name__ == "__main__":
    main()