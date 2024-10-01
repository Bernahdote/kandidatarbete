import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from getStockData import fetch_stock_data
from getTrendsData import fetch_google_trends_data
import statsmodels.api as sm
from statsmodels.stats.stattools import jarque_bera
import warnings
import numpy as np

warnings.filterwarnings("ignore")

# Parameters
symbol = 'XRP-USD'
start_date = '2023-11-17'
end_date = '2024-05-17'
keyword = 'buyxrp'

# Fetch stock and trends data
log_volume_series, trading_dates = fetch_stock_data(symbol, start_date, end_date)
log_interest_series = fetch_google_trends_data(keyword, trading_dates)


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
    X_lagged = sm.add_constant(X.shift(x_order).dropna())  # Lag and align X
    y_aligned = y[X_lagged.index]  # Align y with lagged X

    model_arx = sm.tsa.SARIMAX(y_aligned, exog=X_lagged, order=(ar_order, 0, 0))
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
    result = model_armax.fit()

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
    """Plot cross-correlation function between two series."""
    plt.figure(figsize=(10, 6))
    plt.xcorr(series_x.dropna(), series_y.dropna(), maxlags=lags)
    plt.title('Cross-correlation Function')
    plt.xlabel('Lags')
    plt.ylabel('Cross-correlation')
    plt.show()

 # Plot model predictions vs actual values
def plot_predictions(actual, predicted, title):
    """Plot predicted values against actual values."""
    plt.figure(figsize=(10, 6))
    plt.plot(actual.index, actual, label='Actual Differentiated Volume')
    plt.plot(predicted.index, predicted, label='Predicted Volume', linestyle='--')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Differentiated Volume')
    plt.legend()
    plt.show()

# Perform Jarque-Bera test and plot p-value
def perform_jarque_bera_test(residuals, model_name):
    """Perform Jarque-Bera test and print p-value."""
    jb_stat, jb_pvalue, skew, kurtosis = jarque_bera(residuals.dropna())
    print(f"Jarque-Bera test p-value for {model_name} model residuals: {jb_pvalue}")

if __name__ == '__main__':
    # Plot correlations for volume and interest series
    plot_pacf_correlation(log_volume_series, "log Volume")
    plot_acf_correlation(log_interest_series, "log Interest")
    plot_pacf_correlation(log_interest_series, "log Interest")

    # Initial ARX model fitting
    print("Fitting initial ARX model...")
    # You can choose initial lags based on previous knowledge or set to default
    initial_ar_order = int(input("Enter initial AR order for ARX model (integer >= 0): "))
    initial_x_order = int(input("Enter initial exogenous variable lag order for ARX model (integer >= 0): "))
    result_arx, residuals_arx = fit_arx_model(log_volume_series, log_interest_series, initial_ar_order, initial_x_order)
    plot_acf_correlation(residuals_arx, "ARX")
    perform_jarque_bera_test(residuals_arx, "ARX")

    # Start iterative ARMAX model fitting
    while True:
        print("\n--- Iterative ARMAX Model Fitting ---\n")
        # Plot residuals' ACF, PACF, and cross-correlation
        plot_acf_correlation(residuals_arx, "ARX Residuals")
        plot_ccf(residuals_arx, log_interest_series.loc[residuals_arx.index], lags=40)

        # Ask user to input model parameters
        print("Please input the ARMAX model parameters:")
        ar_order = int(input("Enter AR order (integer >= 0): "))
        ma_order = int(input("Enter MA order (integer >= 0): "))
        x_order = int(input("Enter Exogenous variable lag order (integer >= 0): "))

        # Fit ARMAX model
        result_armax, residuals_armax = fit_armax_model(log_volume_series, log_interest_series, ar_order, ma_order, x_order)
        plot_acf_correlation(residuals_armax, "ARMAX")
        perform_jarque_bera_test(residuals_armax, "ARMAX")

        predictions_armax = result_armax.predict()
        actual_aligned = log_volume_series.loc[predictions_armax.index]
        plot_predictions(actual_aligned, predictions_armax, "ARMAXzzzz")
        

        # Update residuals for next iteration
        residuals_arx = residuals_armax

        # Ask if user wants to continue or exit
        cont = input("Do you want to try different lag orders for the ARMAX model? (yes/no): ").strip().lower()
        if cont != 'yes':
            break
