"""
Created on Fri Sep 13 15:35:04 2024

@author: caspar
"""

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from getStockData import fetch_stock_data
from getTrendsData import fetch_google_trends_data
from statsmodels.tsa.stattools import grangercausalitytests
import statsmodels.api as sm
#from sklearn.preprocessing import StandardScaler

# Set the stock symbol and date range
symbol = 'XRP-USD'
start_date = '2023-11-17'
end_date = '2024-05-17'

keyword = 'buyxrp'

# Fetch stock and trends data
diff_volume_series, trading_dates = fetch_stock_data(symbol, start_date, end_date)
diff_interest_series = fetch_google_trends_data(keyword, trading_dates)

# Scale time series -1,1
#scaler = StandardScaler()
#diff_volume_series_scaled = scaler.fit_transform(diff_volume_series.values.reshape(-1, 1))
#diff_interest_series_scaled = scaler.fit_transform(diff_interest_series.values.reshape(-1, 1))


# Plot ACF for volume series with confidence intervals
plt.figure(figsize=(10, 6))
plot_acf(diff_volume_series, lags=40)
plt.title("ACF Plot for Volume Series")
plt.show()

# Plot PACF for volume series with confidence intervals
plt.figure(figsize=(10, 6))
plot_pacf(diff_volume_series, lags=40)
plt.title("PACF Plot for Volume Series")
plt.show()

# Plot ACF for trends series with confidence intervals
plt.figure(figsize=(10, 6))
plot_acf(diff_interest_series, lags=40)
plt.title("ACF Plot for Interest Series")
plt.show()

# Plot PACF for trends series with confidence intervals
plt.figure(figsize=(10, 6))
plot_pacf(diff_interest_series, lags=40)
plt.title("PACF Plot for Interest Series")
plt.show()

# Function to fit ARX model using SARIMAX
# Function to fit ARX model using SARIMAX
def fit_arx_model(y, X, ar_order, x_order):
    
    # Create lagged versions of the exogenous variable (X)
    X_lagged = sm.add_constant(X.shift(x_order).dropna())  # Add constant and shift exogenous data for lag

    # Align y with X_lagged (matching dates)
    y_aligned = y[X_lagged.index]

    # Fit the ARX model (without MA part, so order (ar_order, 0))
    model_arx = sm.tsa.SARIMAX(y_aligned, exog=X_lagged, order=(ar_order, 0, 0))
    result = model_arx.fit()

    # Get the residuals
    residuals = result.resid

    # Print summary of the model
    print(result.summary())

    # Plot residuals to check for remaining autocorrelation
    plt.figure(figsize=(10, 6))
    plt.plot(residuals)
    plt.title("Residuals of the ARX Model")
    plt.xlabel("Date")
    plt.ylabel("Residuals")
    plt.show()

    return result, residuals

# Function to fit ARMAX model using SARIMAX
def fit_armax_model(y, X, ar_order, ma_order, x_order):
    # Create lagged versions of the exogenous variable (X)
    X_lagged = sm.add_constant(X.shift(x_order).dropna())  # Add constant and shift exogenous data for lag
    y_aligned = y[X_lagged.index]  # Align y with X_lagged (matching dates)

    # Fit the ARMAX model (AR and MA parts included, with exogenous variables)
    model_armax = sm.tsa.SARIMAX(y_aligned, exog=X_lagged, order=(ar_order, 0, ma_order))
    result = model_armax.fit()

    # Get the residuals
    residuals = result.resid

    # Print summary of the model
    print(result.summary())

    # Plot residuals to check for remaining autocorrelation
    plt.figure(figsize=(10, 6))
    plt.plot(residuals)
    plt.title("Residuals of the ARMAX Model")
    plt.xlabel("Date")
    plt.ylabel("Residuals")
    plt.show()

    return result, residuals

# Example usage: fitting ARX model
if __name__ == '__main__':
    ar_order = 5  # Example AR lag order from PACF
    ma_order = 2  # Example MA lag order from ACF
    x_order = 1  # Example X lag order

    # Fit ARX model first
    result_arx, residuals_arx = fit_arx_model(diff_volume_series, diff_interest_series, ar_order, x_order)

    # Print ACF plot for residuals from ARX to check for autocorrelation
    plt.figure(figsize=(10, 6))
    plot_acf(residuals_arx, lags=40)
    plt.title("ACF Plot for Residuals (ARX)")
    plt.show()

    # Now fit ARMAX model
    result_armax, residuals_armax = fit_armax_model(diff_volume_series, diff_interest_series, ar_order, ma_order, x_order)

    # Print ACF plot for residuals from ARMAX to check for remaining autocorrelation
    plt.figure(figsize=(10, 6))
    plot_acf(residuals_armax, lags=40)
    plt.title("ACF Plot for Residuals (ARMAX)")
    plt.show()

    # Generate and plot predictions for the ARMAX model
predictions_armax = result_armax.predict()

# Align actual values with the predicted values for plotting
actual_aligned = diff_volume_series.loc[predictions_armax.index]

# Plot the predictions against actual values
plt.figure(figsize=(10, 6))
plt.plot(actual_aligned.index, actual_aligned, label='Actual Differentiated Volume')
plt.plot(actual_aligned.index, predictions_armax, label='Predicted Volume (ARMAX)', linestyle='--')
plt.title('ARMAX Model Predictions vs Actual Differentiated Volume')
plt.xlabel('Date')
plt.ylabel('Differentiated Volume')
plt.legend()
plt.show()
