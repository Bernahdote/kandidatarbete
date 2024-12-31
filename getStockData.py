import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller
from scipy.stats import zscore



def detect_and_handle_outliers(series, threshold=3):
    """
    Detect and handle outliers in a time series using Z-score method.
    Points with Z-score greater than the threshold will be considered as outliers.
    Returns the cleaned series.
    """
    # Compute Z-scores
    z_scores = zscore(series)

    # Identify outliers where the Z-score is greater than the threshold
    outliers = np.abs(z_scores) > threshold

    # Handle outliers (set outliers to NaN, then fill them with linear interpolation)
    series[outliers] = np.nan
    series = series.interpolate()  # Interpolate NaN values

    return series

def fetch_stock_data(symbol, start_date, end_date):
    """
    Fetches stock data from Yahoo Finance, log-transforms, and performs ADF test.
    """
    # Download the stock data with daily intervals
    stockData = yf.download(symbol, start=start_date, end=end_date, interval='1d')

    # Select the 'Volume' column as a series
    volume_series = stockData['Volume']

    # Log-transform series
    small_value = 1e-5
    volume_series = volume_series.replace(0, small_value)
    log_volume_series = np.log(volume_series / volume_series.shift(1))
    log_volume_series = log_volume_series.dropna()

    # Detect and handle outliers
    log_volume_series_clean = detect_and_handle_outliers(log_volume_series)

    print("Performing ADF test on log-transformed Google Trends series:")
    result = adfuller(log_volume_series_clean)
    print(f"p-value: {result[1]}")
    if result[1] < 0.05:
        print("The Volume series is stationary (reject the null hypothesis).")
    else:
        print("The Volume series is not stationary (fail to reject the null hypothesis).")

    # Return the cleaned log volume series and the trading dates
    return log_volume_series_clean, stockData.index