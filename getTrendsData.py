import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot = False

def detect_and_handle_outliers(series, method="zscore", threshold=3):
    """
    Detect and handle outliers using Z-score or IQR methods.
    Returns a cleaned series without outliers.
    """
    if method == "zscore":
        z_scores = np.abs(stats.zscore(series))
        series_clean = series[z_scores < threshold]  # Removing outliers based on Z-score
    elif method == "iqr":
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        series_clean = series[~((series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR)))]  # Removing outliers based on IQR
    else:
        raise ValueError("Unsupported method for outlier detection")
    
    return series_clean

def fetch_google_trends_data(keyword, trading_dates, weekly):
    """
    Fetches Google Trends data, applies log-transform, performs ADF test, and handles outliers.
    Supports daily or weekly data download based on the boolean flag.
    """
    # Dynamically create the file name using the keyword
    file_name = f"{keyword}_trends.csv"
    folder_path = os.path.expanduser('./trends_data')
    file_path = os.path.join(folder_path, file_name)

    # Load the CSV file
    trendsData = pd.read_csv(file_path, skiprows=3, delimiter=';', names=['Dag', 'Interest'], parse_dates=['Dag'])

    if not weekly:
        # Filter out the weekend days (Mon-Fri only)
        #trendsData['Weekday'] = trendsData['Dag'].dt.weekday
        #trendsData = trendsData[trendsData['Weekday'] < 5]
        #trendsData = trendsData.drop(columns=['Weekday'])
        # Filter Google Trends data to match the available trading dates
        trendsData = trendsData[trendsData['Dag'].isin(trading_dates)]
    else:
        # If weekly, we assume the data can be downloaded in weekly format or resample it to weekly
        trendsData = trendsData.set_index('Dag').resample('W').mean().reset_index()

    # Select the 'Interest' column and apply log transformation
    interest_series = trendsData.set_index('Dag')['Interest']
    small_value = 1e-5
    interest_series = interest_series.replace(0, small_value)
    log_interest_series = np.log(interest_series / interest_series.shift(1))
    log_interest_series = log_interest_series.dropna()

    # Detect and handle outliers in the log-transformed series
    log_interest_series_clean = detect_and_handle_outliers(log_interest_series, method="zscore", threshold=3)

    # Perform the ADF test to check for stationarity
    print("Performing ADF test on log-transformed Google Trends series:")
    result = adfuller(log_interest_series_clean)
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    for key, value in result[4].items():
        print(f"Critical Value {key}: {value}")

    if result[1] < 0.05:
        print("The time series is stationary (reject the null hypothesis).")
    else:
        print("The time series is not stationary (fail to reject the null hypothesis).")

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(interest_series.index, interest_series)
        plt.title(f' {"Weekly" if weekly else "Daily"} Google Trends Data')
        plt.xlabel('Date')
        plt.ylabel(f'Search Interest for {keyword}')
        plt.show()

        # Plot the log-transformed data
        plt.figure(figsize=(10, 6))
        plt.plot(log_interest_series.index, log_interest_series)
        plt.title(f'Log-Transformed {"Weekly" if weekly else "Daily"} Data (Before Outlier Handling)')
        plt.xlabel('Date')
        plt.ylabel(f'Log-Transformed Search Interest for {keyword}')
        plt.show()

        # Plot the cleaned log-transformed data
        plt.figure(figsize=(10, 6))
        plt.plot(log_interest_series_clean.index, log_interest_series_clean)
        plt.title(f'Cleaned Log-Transformed {"Weekly" if weekly else "Daily"} Data')
        plt.xlabel('Date')
        plt.ylabel(f'Cleaned Log-Transformed Search Interest for {keyword}')
        plt.show()

    return log_interest_series_clean
