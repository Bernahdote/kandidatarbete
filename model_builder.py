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

# Set the stock symbol and date range
symbol = 'SPY'
start_date = '2023-11-17'
end_date = '2024-05-17'

keyword = 'snp500'

# Fetch stock and trends data
diff_volume_series, trading_dates = fetch_stock_data(symbol, start_date, end_date)
diff_interest_series = fetch_google_trends_data(keyword, trading_dates)

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
def fit_arx_model(y, X, ar_order, x_order):
    """
    Fits an ARX model with the specified AR order and exogenous lags.

    Parameters:
    y : pandas Series
        The dependent time series (e.g., differenced volume series).
    X : pandas Series
        The exogenous time series (e.g., differenced interest series).
    ar_order : int
        The number of lags to include in the AR (autoregressive) part.
    x_order : int
        The number of lags to include for the exogenous variable.

    Returns:
    result : ARMAXResults
        The fitted ARX model.
    residuals : pandas Series
        The residuals from the fitted ARX model.
    """

    # Create lagged versions of the exogenous variable (X)
    X_lagged = sm.add_constant(X.shift(x_order).dropna())  # Add constant and shift exogenous data for lag
    y_aligned = y[X_lagged.index]  # Align y with X_lagged (matching dates)

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

#

if __name__ == '__main__':
#Example usage: fitting ARX model
    ar_order = 3  # Example AR lag order
    x_order = 5  # Example X lag order

    result, residuals = fit_arx_model(diff_volume_series, diff_interest_series, ar_order, x_order)   

 

# Define the ARMAX model
# Replace p and q with the number of AR and MA lags determined from the ACF/PACF plots
   # model = sm.tsa.ARMA(endog=y, exog=X, order=(p, q))  # Replace p and q with actual lags

# Fit the model
   # results = model.fit()

# Print the model summary
  #  print(results.summary())