import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from getStockData import fetch_stock_data
from getTrendsData import fetch_google_trends_data
import statsmodels.api as sm
from statsmodels.stats.stattools import jarque_bera
import warnings
import numpy as np
from statsmodels.tsa.stattools import ccf
from sklearn.metrics import mean_squared_error
#from pmdarima import auto_arima

warnings.filterwarnings("ignore")

# Parameters
weekly = False  # Set this to True for weekly, False for daily

symbol = 'SHIB-USD'
start_date = '2023-11-17'
end_date = '2024-05-17'
keyword = 'shibainu'

# Fetch stock and trends data
log_volume_series, trading_dates = fetch_stock_data(symbol, start_date, end_date, weekly=weekly)
log_interest_series = fetch_google_trends_data(keyword, trading_dates, weekly=weekly)

def split_data(y, X, train_size=0.8):
    """Splits y and X into aligned train and test sets."""
    # Find the number of observations for training
    train_len = int(len(y) * train_size)

    # Align both y and X using the same index
    common_index = y.index.intersection(X.index)
    y_aligned = y.loc[common_index]
    X_aligned = X.loc[common_index]

    # Now split the aligned data into train/test
    y_train, y_test = y_aligned[:train_len], y_aligned[train_len:]
    X_train, X_test = X_aligned[:train_len], X_aligned[train_len:]

    return y_train, y_test, X_train, X_test

# Split data into 80% training and 20% testing
log_volume_train, log_volume_test, log_interest_train, log_interest_test = split_data(log_volume_series, log_interest_series)

# Plot PACF
def plot_pacf_correlation(series, title_prefix):
    plt.figure(figsize=(10, 6))
    plot_pacf(series.dropna(), lags=40)
    plt.title(f"PACF Plot for {title_prefix} Series")
    plt.show()

# Plot ACF 
def plot_acf_correlation(series, title_prefix):
    plt.figure(figsize=(10, 6))
    plot_acf(series.dropna(), lags=40)
    plt.title(f"ACF Plot for {title_prefix} Series")
    plt.show()

# Fit ARX model
def fit_arx_model(y, X, ar_order, x_order):
    """Fit ARX model using SARIMAX."""
    # Align the indices of X and y
    X_aligned, y_aligned = X.align(y, join='inner')
    
    # Shift the exogenous variable and add a constant
    X_lagged = sm.add_constant(X_aligned.shift(x_order).dropna())  # Lag and align X
    
    y_lagged = y_aligned.loc[X_lagged.index]  # Align y with lagged X
    
    model_arx = sm.tsa.SARIMAX(y_lagged, exog=X_lagged, order=(ar_order, 0, 0))
    result = model_arx.fit()

    print(result.summary())

    # Plot residuals
    residuals = result.resid
    plt.figure(figsize=(10, 6))
    plt.plot(residuals)
    plt.title("Residuals of the ARX Model")
    plt.xlabel("Date")
    plt.ylabel("Residuals")
    plt.show()

    return result, residuals


# Fit ARMAX model
def fit_armax_model(y, X, ar_order, ma_order, x_order):
    """Fit ARMAX model using SARIMAX."""
    X_lagged = sm.add_constant(X.shift(x_order).dropna())  # Lag and align X
    y_aligned = y[X_lagged.index]  # Align y with lagged X

    model_armax = sm.tsa.SARIMAX(y_aligned, exog=X_lagged, order=(ar_order, 0, ma_order))
    result = model_armax.fit(maxiter=1000)

    print(result.summary())

    # Plot residuals
    residuals = result.resid
    plt.figure(figsize=(10, 6))
    plt.plot(residuals)
    plt.title("Residuals of the ARMAX Model")
    plt.xlabel("Date")
    plt.ylabel("Residuals")
    plt.show()

    return result, residuals

# Plot cross-correlation function
def plot_ccf(series_x, series_y, lags=40):
    cross_corr = ccf(series_x.dropna(), series_y.dropna())[:lags + 1]
    conf_level = 1.96 / np.sqrt(len(series_x))  # 95% confidence interval
    plt.figure(figsize=(10, 6))
    plt.stem(range(lags + 1), cross_corr)
    plt.axhline(y=conf_level, linestyle='--', color='r', label=f"95% Confidence (+/- {conf_level:.2f})")
    plt.axhline(y=-conf_level, linestyle='--', color='r')
    plt.axhline(y=0, color='black', lw=1)
    significant_lags = np.where(np.abs(cross_corr) > conf_level)[0]
    if len(significant_lags) > 0:
        plt.scatter(significant_lags, cross_corr[significant_lags], color='red', zorder=5, label='Significant Lags')
    plt.title('Cross-correlation Function with Confidence Intervals')
    plt.xlabel('Lags')
    plt.ylabel('Cross-correlation')
    plt.legend()
    plt.show()

# Perform Jarque-Bera test
def perform_jarque_bera_test(residuals, model_name):
    jb_stat, jb_pvalue, skew, kurtosis = jarque_bera(residuals.dropna())
    print(f"Jarque-Bera test p-value for {model_name} model residuals: {jb_pvalue}")

# Plot predictions vs actual
def plot_predictions(actual, predicted, title):
    # Align predictions and actual test data by intersecting their indices
    common_test_index = predictions_test.index.intersection(log_volume_test.index)

    # Now align both the predicted and actual data
    actual_test_aligned = log_volume_test.loc[common_test_index]
    predictions_test_aligned = predictions_test.loc[common_test_index]

    #Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(actual_test_aligned.index, actual_test_aligned, label='Actual Volume')
    plt.plot(predictions_test_aligned.index, predictions_test_aligned, label='Predicted Volume', linestyle='--')
    plt.title('ARMAX Model Predictions VS Actual Traded Volume')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # Plot correlations for volume and interest series
    plot_pacf_correlation(log_volume_train, "log Volume (Training)")
    #plot_acf_correlation(log_interest_train, "log Interest (Training)")

    # Initial ARX model fitting
    print("Fitting initial ARX model...")
    initial_ar_order = int(input("Enter initial AR order for ARX model: "))
    initial_x_order = int(input("Enter Exogenous variable lag order for ARX model: "))
    result_arx, residuals_arx = fit_arx_model(log_volume_train, log_interest_train, initial_ar_order, initial_x_order)

    plot_acf_correlation(residuals_arx, "ARX")
    perform_jarque_bera_test(residuals_arx, "ARX")

    # Start iterative ARMAX model fitting
    while True:
        print("\n--- Iterative ARMAX Model Fitting ---\n")
        plot_ccf(residuals_arx, log_interest_train.loc[residuals_arx.index], lags=40)

        # Ask user to input model parameters
        print("Please input the ARMAX model parameters:")
        ar_order = int(input("Enter AR order (integer >= 0): "))
        ma_order = int(input("Enter MA order (integer >= 0): "))
        x_order = int(input("Enter Exogenous variable lag order (integer >= 0): "))

        # Fit ARMAX model on training data
        result_armax, residuals_armax = fit_armax_model(log_volume_train, log_interest_train, ar_order, ma_order, x_order)
        plot_acf_correlation(residuals_armax, "ARMAX")
        perform_jarque_bera_test(residuals_armax, "ARMAX")

        # Prepare to store the predictions
        predictions_test = []

        # Rolling forecast
        for i in range(len(log_interest_test)):
            # Combine past data with test predictions
            past_interest = pd.concat([log_interest_train, log_interest_test[:i]])
            past_volume = pd.concat([log_volume_train, pd.Series(predictions_test)])

            # Align the exogenous data with the correct lag
            exog_current = sm.add_constant(past_interest.shift(x_order).dropna())
            exog_current = exog_current.iloc[-1:].values.reshape(1, -1)  # Get the last row

            # Forecast the next step using the model
            pred = result_armax.get_forecast(steps=1, exog=exog_current).predicted_mean
            predictions_test.append(pred.iloc[0])

        # Convert predictions to a series with the test index
        predictions_test_series = pd.Series(predictions_test, index=log_interest_test.index[:len(predictions_test)])

        # Compare predictions to actual values and calculate MSE
        mse = mean_squared_error(log_volume_test, predictions_test_series)
        print(f"Mean Squared Error (MSE) on the test set: {mse}")

        # Calculate correct rise/fall predictions
        actual_changes = log_volume_test.diff().dropna()
        predicted_changes = predictions_test_series.diff().dropna()

        correct_predictions = (np.sign(actual_changes) == np.sign(predicted_changes)).sum()
        total_predictions = len(predicted_changes)

        print(f"Correct Rise/Fall Predictions: {correct_predictions}/{total_predictions}")
        accuracy = correct_predictions / total_predictions * 100
        print(f"Accuracy of predicting rise/fall: {accuracy:.2f}%")

        # Plot predictions vs actual
        plt.figure(figsize=(10, 6))
        plt.plot(log_volume_test.index, log_volume_test, label='Actual Volume')
        plt.plot(predictions_test_series.index, predictions_test_series, label='Predicted Volume', linestyle='--')
        plt.title('Rolling Forecast: ARMAX Predictions VS Actual Traded Volume')
        plt.xlabel('Date')
        plt.ylabel('Volume')
        plt.legend()
        plt.show()

        # Update residuals for next iteration
        residuals_arx = residuals_armax

        # Ask if user wants to continue or exit
        cont = input("Do you want to try different lag orders for the ARMAX model? (yes/no): ").strip().lower()
        if cont != 'yes':
            break