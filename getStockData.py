import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from getTrendsData import fetch_google_trends_data
from statsmodels.tsa.stattools import adfuller
from scipy.stats import zscore
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


plot = False

def fetch_stock_data(symbol, start_date, end_date):
    """
    Fetches stock data and returns the log-transformed volume series and trading dates.
    """

    # Fetch data
    stock_data = yf.download(symbol, start=start_date, end=end_date)

    stock_data = stock_data[['Volume']]

    # Log-transform the volume data
    volume_series = stock_data['Volume']
    small_value = 1e-5
    volume_series = volume_series.replace(0, small_value)
    log_volume_series = np.log(volume_series / volume_series.shift(1))
    log_volume_series = log_volume_series.dropna()


    # Extract trading dates
    trading_dates = stock_data.index

    return log_volume_series, trading_dates