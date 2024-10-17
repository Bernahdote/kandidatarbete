import os
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt


def detect_and_handle_outliers(series, method="zscore", threshold=3):
    if method == "zscore":
        from scipy.stats import zscore
        z_scores = zscore(series)
        abs_z_scores = np.abs(z_scores)
        filtered_entries = abs_z_scores < threshold
        return series[filtered_entries]
    

def fetch_google_trends_data(keyword, trading_dates, start_date1, end_date1, start_date2, end_date2, weekly=False, plot=False):
    """
    Fetches Google Trends data from CSV files for two intervals, normalizes them using median,
    applies log-transform, performs ADF test, and handles outliers.
    """
    # Construct file paths
    folder_path = os.path.expanduser('./trends_data')
    file_name1 = f"{keyword}1_trends.csv"
    file_name2 = f"{keyword}2_trends.csv"
    file_path1 = os.path.join(folder_path, file_name1)
    file_path2 = os.path.join(folder_path, file_name2)

    # Load the CSV files
    trendsData1 = pd.read_csv(file_path1, skiprows=3, delimiter=';', names=['Dag', 'Interest'], parse_dates=['Dag'])
    trendsData2 = pd.read_csv(file_path2, skiprows=3, delimiter=';', names=['Dag', 'Interest'], parse_dates=['Dag'])

    # Ensure 'Dag' is in datetime format and set as index
    trendsData1['Dag'] = pd.to_datetime(trendsData1['Dag'])
    trendsData2['Dag'] = pd.to_datetime(trendsData2['Dag'])
    trendsData1.set_index('Dag', inplace=True)
    trendsData2.set_index('Dag', inplace=True)

    # Sort the DataFrames
    trendsData1.sort_index(inplace=True)
    trendsData2.sort_index(inplace=True)

    # Filter data to the specified intervals
    trendsData1 = trendsData1[start_date1:end_date1]
    trendsData2 = trendsData2[start_date2:end_date2]

    # Identify overlapping period
    overlap_start = max(trendsData1.index.min(), trendsData2.index.min())
    overlap_end = min(trendsData1.index.max(), trendsData2.index.max())

    # Extract overlapping data
    overlap_data1 = trendsData1.loc[overlap_start:overlap_end]
    overlap_data2 = trendsData2.loc[overlap_start:overlap_end]

    # Check if overlap exists
    if overlap_data1.empty or overlap_data2.empty:
        raise ValueError("No overlapping data found between the two intervals.")

    # Calculate scaling factor using median
    median_overlap1 = overlap_data1['Interest'].median()
    median_overlap2 = overlap_data2['Interest'].median()

    if median_overlap2 == 0:
        scaling_factor = 1
    else:
        scaling_factor = median_overlap1 / median_overlap2

    # Adjust the second dataset
    trendsData2['Interest'] = trendsData2['Interest'] * scaling_factor

    # Combine datasets
    combined_data = pd.concat([trendsData1, trendsData2])
    combined_data = combined_data[~combined_data.index.duplicated(keep='first')]
    combined_data.sort_index(inplace=True)

    print(f"Combined data before reindexing: {len(combined_data)} entries")

    if not weekly:
        # Ensure trading_dates are in datetime format
        trading_dates = pd.to_datetime(trading_dates)
        # Normalize dates to remove time components
        combined_data.index = combined_data.index.normalize()
        trading_dates = trading_dates.normalize()

        # Combine the dates from trading_dates and combined_data
        all_dates = combined_data.index.union(trading_dates)
        combined_data = combined_data.reindex(all_dates)

        # Interpolate missing 'Interest' values
        combined_data['Interest'].interpolate(method='time', inplace=True)

        print(f"Combined data after reindexing: {len(combined_data)} entries")

        # Now we can filter to trading_dates if required
        combined_data = combined_data.loc[trading_dates]
    else:
        # Resample to weekly frequency
        combined_data = combined_data.resample('W').mean()

    # Replace zeros with a small value to avoid log(0)
    small_value = 1e-5
    combined_data['Interest'] = combined_data['Interest'].replace(0, small_value)

    # Calculate log returns
    interest_series = combined_data['Interest']
    log_interest_series = np.log(interest_series / interest_series.shift(1))
    log_interest_series = log_interest_series.dropna()

    # Detect and handle outliers
    log_interest_series_clean = detect_and_handle_outliers(log_interest_series, method="zscore", threshold=3)

    # Perform the ADF test
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

    # Plotting
    if plot:
        # Plot the interest series
        plt.figure(figsize=(12, 6))
        plt.plot(interest_series.index, interest_series)
        plt.title(f'{"Weekly" if weekly else "Daily"} Google Trends Data for "{keyword}"')
        plt.xlabel('Date')
        plt.ylabel('Search Interest')
        plt.show()
        plt.close()

        # Plot the log-transformed data
        plt.figure(figsize=(12, 6))
        plt.plot(log_interest_series.index, log_interest_series)
        plt.title(f'Log-Transformed {"Weekly" if weekly else "Daily"} Data (Before Outlier Handling)')
        plt.xlabel('Date')
        plt.ylabel('Log Returns')
        plt.show()
        plt.close()

        # Plot the cleaned log-transformed data
        plt.figure(figsize=(12, 6))
        plt.plot(log_interest_series_clean.index, log_interest_series_clean)
        plt.title(f'Cleaned Log-Transformed {"Weekly" if weekly else "Daily"} Data')
        plt.xlabel('Date')
        plt.ylabel('Log Returns')
        plt.show()
        plt.close()

    return log_interest_series_clean
