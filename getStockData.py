import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from getTrendsData import fetch_google_trends_data

def fetch_stock_data(symbol, start_date, end_date):

    # Download the stock data with daily intervals
    stockData = yf.download(symbol, start=start_date, end=end_date, interval='1d')
   
    # Select the 'Volume' column as a series
    volume_series = stockData['Volume']

    # Apply differencing (to remove trend)
    diff_volume_series = volume_series.diff().dropna()

    # Plot the original data using the index for dates
    plt.figure(figsize=(10, 6))
    plt.plot(stockData.index, volume_series)
    plt.title('Original Data')
    plt.xlabel('Date')
    plt.ylabel(f'Volume for {symbol}')
    plt.show()

    # Plot the differenced data
    plt.figure(figsize=(10, 6))
    plt.plot(stockData.index[1:], diff_volume_series)
    plt.title('Differenced Data')
    plt.xlabel('Date')
    plt.ylabel(f'Differenced Volume for {symbol}')
    plt.show()

    # Return the differenced series and the trading dates
    return diff_volume_series, stockData.index