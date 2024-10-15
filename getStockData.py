import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from getTrendsData import fetch_google_trends_data
from statsmodels.tsa.stattools import adfuller
from scipy.stats import zscore
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


plot = False

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
    
    # Alternatively, we could remove outliers instead of interpolating
    # series = series[~outliers]
    
    return series

def fetch_stock_data(symbol, start_date, end_date, weekly):
    """
    Fetches stock data from Yahoo Finance, log-transforms, and performs ADF test.
    """
    # Download the stock data with daily intervals
    stockData = yf.download(symbol, start=start_date, end=end_date, interval='1d')

    # If weekly flag is set, resample to weekly and sum the volume
    if weekly:
        stockData = stockData.resample('W').sum()
    
    # Select the 'Volume' column as a series
    volume_series = stockData['Volume']

    # Log-transform series
    small_value = 1e-5
    volume_series = volume_series.replace(0, small_value)
    log_volume_series = np.log(volume_series / volume_series.shift(1))
    log_volume_series = log_volume_series.dropna()

    # Detect and handle outliers
    log_volume_series_cleaned = detect_and_handle_outliers(log_volume_series)

    # Perform the ADF test to check for stationarity
    #print("Performing ADF test on log-transformed series:")
    #result = adfuller(log_volume_series_cleaned)
    #print(f"ADF Statistic: {result[0]}")
    #print(f"p-value: {result[1]}")
    #for key, value in result[4].items():
    #   print(f"Critical Value {key}: {value}")

    #if result[1] < 0.05:
     #   print("The time series is stationary (reject the null hypothesis).")
    #else:
     #   print("The time series is not stationary (fail to reject the null hypothesis).")

    if plot:
        # Plot the original volume data
        plt.figure(figsize=(10, 6))
        plt.plot(stockData.index, volume_series)
        plt.title('Original Volume Data')
        plt.xlabel('Date')
        plt.ylabel(f'Volume for {symbol}')
        plt.show()

        # Plot the logged data (with outliers)
        plt.figure(figsize=(10, 6))
        plt.plot(log_volume_series.index, log_volume_series)
        plt.title('Logged Volume Data (with outliers)')
        plt.xlabel('Date')
        plt.ylabel(f'Logged Volume for {symbol}')
        plt.show()

        # Plot the cleaned log volume data (after handling outliers)
        plt.figure(figsize=(10, 6))
        plt.plot(log_volume_series_cleaned.index, log_volume_series_cleaned)
        plt.title('Cleaned Logged Volume Data (outliers handled)')
        plt.xlabel('Date')
        plt.ylabel(f'Cleaned Logged Volume for {symbol}')
        plt.show()

    # Return the cleaned log volume series and the trading dates
    return log_volume_series_cleaned, stockData.index


