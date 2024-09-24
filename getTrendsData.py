import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def fetch_google_trends_data(keyword, trading_dates):

    # Dynamically create the file name using the keyword
    file_name = f"{keyword}_trends.csv"

    # Define the folder path
    folder_path = os.path.expanduser('./trends_data')

    # Combine the folder path and file name to get the full file path
    file_path = os.path.join(folder_path, file_name)

    # Load the CSV file, skipping the first three rows, and specifying the delimiter as ';'
    trendsData = pd.read_csv(f'{folder_path}/{file_name}', skiprows=3, delimiter=';', names=['Dag', 'Interest'], parse_dates=['Dag'])

    # Filter out the weekend days (Saturday = 5, Sunday = 6)
    trendsData['Weekday'] = trendsData['Dag'].dt.weekday  # Add a new column for the weekday (0 = Monday, ..., 6 = Sunday)
    trendsData_weekdays = trendsData[trendsData['Weekday'] < 5]  # Keep only rows where the weekday is less than 5 (Mon-Fri)

    # Drop the 'Weekday' column since it's no longer needed
    trendsData_weekdays = trendsData_weekdays.drop(columns=['Weekday'])

    # Filter Google Trends data to match the available trading dates
    trendsData_weekdays = trendsData_weekdays[trendsData_weekdays['Dag'].isin(trading_dates)]

    # Select the correct column for the time series data (Interest)
    series = trendsData_weekdays.set_index('Dag')['Interest']  # Set 'Dag' as index to align the differenced data properly

    # 1. Plot the original data (weekdays only, matched with trading days)
    plt.figure(figsize=(10, 6))
    plt.plot(series.index, series)
    plt.title('Original Data (Matched with Trading Days)')
    plt.xlabel('Date')
    plt.ylabel(f'Search Interest for {keyword}')
    plt.show()

    # 3. Apply differencing (to remove trend)
    scaler = StandardScaler()
    diff_interest_series = series.diff().dropna()  # First difference
    diff_interest_series_scaled = scaler.transform(diff_interest_series.values.reshape(-1, 1))

    # Plot the differenced data (matched with trading days)
    plt.figure(figsize=(10, 6))
    plt.plot(diff_interest_series.index, diff_interest_series_scaled)
    plt.title('Differenced Data (Matched with Trading Days)')
    plt.xlabel('Date')
    plt.ylabel(f'Differenced Search Interest for {keyword}')
    plt.show()
   
    return diff_interest_series_scaled
