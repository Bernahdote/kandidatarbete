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

    # Return the cleaned log volume series and the trading dates
    return original_volume_series, stockData.index
