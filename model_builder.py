"""
Created on Fri Sep 13 15:35:04 2024

@author: caspar
"""

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from getStockData import fetch_stock_data
from getTrendsData import fetch_google_trends_data
import statsmodels.api as sm
# Parameters
symbol = 'XRP-USD'
start_date = '2023-11-17'
end_date = '2024-05-17'
keyword = 'buyxrp'
ar_order = 5  # Example AR lag order from PACF
ma_order = 2  # Example MA lag order from ACF
x_order = 1  # Example X lag order

# Fetch stock and trends data
diff_volume_series, trading_dates = fetch_stock_data(symbol, start_date, end_date)
diff_interest_series = fetch_google_trends_data(keyword, trading_dates)

# Plot ACF and PACF
def plot_correlations(series, title_prefix):
    """Plot ACF and PACF for a given time series."""
    plt.figure(figsize=(10, 6))
    plot_acf(series, lags=40)
    plt.title(f"ACF Plot for {title_prefix} Series")
    plt.show()

    plt.figure(figsize=(10, 6))
    plot_pacf(series, lags=40)
    plt.title(f"PACF Plot for {title_prefix} Series")
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

# Plot residuals ACF
def plot_residual_acf(residuals, model_name):
    """Plot ACF of model residuals."""
    plt.figure(figsize=(10, 6))
    plot_acf(residuals, lags=40)
    plt.title(f"ACF Plot for Residuals ({model_name})")
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

if __name__ == '__main__':
    # Plot correlations for volume and interest series
    plot_correlations(diff_volume_series, "Volume")
    plot_correlations(diff_interest_series, "Interest")

    # Fit ARX model
    result_arx, residuals_arx = fit_arx_model(diff_volume_series, diff_interest_series, ar_order, x_order)
    plot_residual_acf(residuals_arx, "ARX")

    # Fit ARMAX model the first time
    result_armax, residuals_armax = fit_armax_model(diff_volume_series, diff_interest_series, ar_order, ma_order, x_order)
    plot_residual_acf(residuals_armax, "ARMAX")

    # Plot predictions for ARMAX model
    predictions_armax = result_armax.predict()
    actual_aligned = diff_volume_series.loc[predictions_armax.index]
    plot_predictions(actual_aligned, predictions_armax, "ARMAX Model Predictions vs Actual Differentiated Volume")
