import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore
from statsmodels.tsa.stattools import adfuller



def fetch_stock_data(symbol, start_date, end_date):
    """
    Fetches stock data from Yahoo Finance, log-transforms, and performs ADF test.
    """
    # Download the stock data with daily intervals
    stockData = yf.download(symbol, start=start_date, end=end_date, interval='1d')

    # Select the 'Volume' column as a series
    original_volume_series = stockData['Volume']

    # Now split the aligned data into train/test
    train_len = int(len(original_volume_series) * 0.8)
    y_train = original_volume_series.iloc[:train_len]

    initial_volume = original_volume_series[len(y_train)-1]

    # Log-transform series
    small_value = 1e-5
    original_volume_series = original_volume_series.replace(0, small_value)
    log_volume_series = np.log10(original_volume_series / original_volume_series.shift(1))
    log_volume_series = log_volume_series.dropna()
    

    # Return the cleaned log volume series and the trading dates
    return original_volume_series, stockData.index
