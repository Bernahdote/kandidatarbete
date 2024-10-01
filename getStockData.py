import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from getTrendsData import fetch_google_trends_data

plot = True


def fetch_stock_data(symbol, start_date, end_date):

    # Download the stock data with daily intervals
    stockData = yf.download(symbol, start=start_date, end=end_date, interval='1d')
   
    # Select the 'Volume' column as a series
    volume_series = stockData['Volume']

     # Log-transform series
    small_value = 1e-5

    volume_series = volume_series.replace(0, small_value)

    log_volume_series = np.log(volume_series / volume_series.shift(1))

    log_volume_series = log_volume_series.dropna()

    if plot: 
        # Plot the original data using the index for dates
        plt.figure(figsize=(10, 6))
        plt.plot(stockData.index, volume_series)
        plt.title('Original Data')
        plt.xlabel('Date')
        plt.ylabel(f'Volume for {symbol}')
        plt.show()

        # Plot the logged data
        plt.figure(figsize=(10, 6))
        plt.plot(log_volume_series.index, log_volume_series)
        plt.title('logged Data')
        plt.xlabel('Date')
        plt.ylabel(f'logged Volume for {symbol}')
        plt.show()

    # Return the original volume series and the trading dates
    return log_volume_series, stockData.index

    if plot: 
        # Plot the original data using the index for dates
        plt.figure(figsize=(10, 6))
        plt.plot(stockData.index, volume_series)
        plt.title('Original Data')
        plt.xlabel('Date')
        plt.ylabel(f'Volume for {symbol}')
        plt.show()

        # Plot the differenced data
        plt.figure(figsize=(10, 6))
        plt.plot(stockData.index[1:], volume_series)
        plt.title('Differenced Data')
        plt.xlabel('Date')
        plt.ylabel(f'Differenced Volume for {symbol}')
        plt.show()

    # Return the differenced series and the trading dates
    return log_volume_series, stockData.index