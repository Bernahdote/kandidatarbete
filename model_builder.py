# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statsmodels.stats.stattools import jarque_bera
import warnings
import itertools
from sklearn.metrics import mean_squared_error
from getStockData import fetch_stock_data
from getTrendsData import fetch_google_trends_data

warnings.filterwarnings("ignore")


symbol = 'BTC-USD'
start_date1 = '2023-03-23'
end_date1 = '2023-11-23'
start_date2 = '2023-11-17'
end_date2 = '2024-07-17'


keyword = 'BTC'
folder = 'BITCOIN'


p = range(0, 5)  # AR order
q = range(0, 5)  # MA order 
x_order_range = range(1, 5)  # Exogenous variable lags


def split_data(y, X, train_size=0.8):
    """Splits y and X into aligned train and test sets over the union of their indexes."""
    # Align both y and X over the union of their indexes
    all_index = y.index.union(X.index)
    y_aligned = y.reindex(all_index)
    X_aligned = X.reindex(all_index)

    # Fill missing values (forward-fill or interpolate)
    y_aligned = y_aligned.fillna(method='ffill')
    X_aligned = X_aligned.fillna(method='ffill')

    # Now split the aligned data into train/test
    train_len = int(len(y_aligned) * train_size)
    y_train, y_test = y_aligned.iloc[:train_len], y_aligned.iloc[train_len:]
    X_train, X_test = X_aligned.iloc[:train_len], X_aligned.iloc[train_len:]

    return y_train, y_test, X_train, X_test

def perform_adfuller_test(series, series_name):
    print(f"Performing ADF test on {series_name}:")
    result = adfuller(series.dropna())
    print(f"p-value: {result[1]}")
    if result[1] < 0.05:
        print("The Volume series is stationary (reject the null hypothesis).")
    else:
        print("The time series is non-stationary (fail to reject the null hypothesis).")
    print()


# Fit ARMAX model
def fit_armax_model(y, X, ar_order, ma_order, x_order):
    """Fit ARMAX model using SARIMAX with multiple lags of the exogenous variable."""
    # Create lagged versions of X up to x_order
    X_lags = pd.concat([X.shift(i) for i in range(1, x_order + 1)], axis=1)
    X_lags.columns = [f'X_lag_{i}' for i in range(1, x_order + 1)]

    # Drop rows with NaN values that result from shifting
    X_lags = X_lags.dropna()
    y_aligned = y.loc[X_lags.index]  # Align y with the lagged X data

    # Add a constant term for intercept
    X_lags = sm.add_constant(X_lags)

    # Fit the SARIMAX model with the AR, MA, and exogenous inputs
    try:
        model_armax = sm.tsa.SARIMAX(y_aligned, exog=X_lags, order=(ar_order, 0, ma_order))
        result = model_armax.fit(disp=False)
        aic = result.aic
        # Return the exogenous variable column names
        exog_columns = X_lags.columns.tolist()
        # Check parameter significance
        p_values = result.pvalues
        # Allow models with at most two insignificant parameters (p-value >= 0.05)
        num_insignificant = (p_values >= 0.05).sum()
        if num_insignificant > 2:
            print(f"Model with AR={ar_order}, MA={ma_order}, X_lags={x_order} has more than two insignificant parameters.")
            return None, np.inf, None
    except Exception as e:
        # If the model fails to converge, return None for result and np.inf for AIC
        print(f"Failed to fit model with AR={ar_order}, MA={ma_order}, X_lags={x_order}: {e}")
        result = None
        aic = np.inf
        exog_columns = None

    return result, aic, exog_columns

# Define fit_arma_model function
def fit_arma_model(y, ar_order, ma_order):
    """Fit ARMA model using SARIMAX without exogenous variables."""
    y_aligned = y.dropna()
    # Fit the SARIMAX model with the AR and MA terms
    try:
        model_arma = sm.tsa.SARIMAX(y_aligned, order=(ar_order, 0, ma_order), trend='c')
        result = model_arma.fit(disp=False)
        aic = result.aic
        # Check parameter significance
        p_values = result.pvalues
        # Allow models with at most two insignificant parameters (p-value >= 0.05)
        num_insignificant = (p_values >= 0.05).sum()
        if num_insignificant > 2:
            print(f"Model with AR={ar_order}, MA={ma_order} has more than two insignificant parameters.")
            return None, np.inf
    except Exception as e:
        # If the model fails to converge, return None for result and np.inf for AIC
        print(f"Failed to fit model with AR={ar_order}, MA={ma_order}: {e}")
        result = None
        aic = np.inf

    return result, aic


# Perform Jarque-Bera test
def perform_jarque_bera_test(residuals, model_name):
    jb_stat, jb_pvalue, skew, kurtosis = jarque_bera(residuals.dropna())
    print(f"Jarque-Bera test p-value for {model_name} model residuals: {jb_pvalue}") 
    
if __name__ == '__main__':

    # Fetch stock and trends data
    log_volume_series, trading_dates = fetch_stock_data(symbol, start_date1, end_date2)
    log_interest_series = fetch_google_trends_data(keyword, folder, trading_dates, start_date1, end_date1, start_date2, end_date2, plot=True)

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(log_interest_series.index, log_interest_series, color = 'blue')
    plt.title(f'Log-Transformed Google Trends Data with Outliers Handled for search word {keyword}')
    plt.xlabel('Date')
    plt.ylabel('Log Returns')

    # Second subplot: Log-Transformed Traded Volume Data
    plt.subplot(2, 1, 2)
    plt.plot(log_volume_series.index, log_volume_series, color = 'red')
    plt.title(f'Log-Transformed Traded Volume Data for stock {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Log Returns')

    # Show the combined plot
    plt.tight_layout()
    plt.show()

    # Split data into training and testing sets
    log_volume_train, log_volume_test, log_interest_train, log_interest_test = split_data(log_volume_series, log_interest_series, train_size=0.8)

    # Perform ADF test on the exogenous variable
    perform_adfuller_test(log_interest_train, "log-transformed Google Trends series")

    # Create combinations of p, q, and x_order
    pdq = list(itertools.product(p, q))
    x_orders = list(x_order_range)

    # Initialize variables to store the best model
    best_aic = np.inf
    best_order = None
    best_x_order = None
    best_result = None
    best_exog_columns = None

    # Grid search over p, q, and x_order
    print("Starting grid search over p, q, and x_order allowing up to two insignificant parameters...")
    for order in pdq:
        for x_order in x_orders:
            print(f"Trying AR={order[0]}, MA={order[1]}, X_lags={x_order}")
            result, aic, exog_columns = fit_armax_model(log_volume_train, log_interest_train, order[0], order[1], x_order)
            if result is not None and aic < best_aic:
                best_aic = aic
                best_order = order
                best_x_order = x_order
                best_result = result
                best_exog_columns = exog_columns

    # Check if a best model was found
    if best_result is not None:
        print(f"\nBest ARMAX model found: AR={best_order[0]}, MA={best_order[1]}, X_lags={best_x_order} with AIC={best_aic}")
        print(best_result.summary())

        # Perform residual analysis
        residuals = best_result.resid
        perform_jarque_bera_test(residuals, "Best ARMAX Model")

        # Rolling forecast with confidence intervals
        predictions = []
        lower_bounds = []  # Store lower bounds of confidence intervals
        upper_bounds = []  # Store upper bounds of confidence intervals
        history_y = log_volume_train.copy()
        history_X = log_interest_train.copy()
        test_index = log_volume_test.index

        for t in range(len(log_volume_test)):
            # Prepare exogenous variables for training
            X_lags = pd.concat([history_X.shift(i) for i in range(1, best_x_order + 1)], axis=1)
            X_lags.columns = [f'X_lag_{i}' for i in range(1, best_x_order + 1)]
            X_lags = sm.add_constant(X_lags)
            X_lags = X_lags.dropna()

            # Align endogenous variable
            y_aligned = history_y.loc[X_lags.index]

            # Re-fit the model on the history data
            model = sm.tsa.SARIMAX(y_aligned, exog=X_lags, order=(best_order[0], 0, best_order[1]))
            result = model.filter(best_result.params)

            # Prepare exogenous variables for forecasting
            exog_forecast_lags = []
            for lag in range(1, best_x_order + 1):
                idx = -lag + t + 1
                if idx >= 0:
                    exog_value = log_interest_test.iloc[idx]
                else:
                    exog_value = history_X.iloc[idx]
                exog_forecast_lags.append(exog_value)
            exog_forecast_lags = exog_forecast_lags[::-1]  # Reverse to match the lag order

            # Combine with constant term
            exog_forecast_values = [1] + exog_forecast_lags
            exog_forecast_df = pd.DataFrame([exog_forecast_values], columns=best_exog_columns)

            # Forecast one step ahead, including confidence intervals
            forecast_result = result.get_forecast(steps=1, exog=exog_forecast_df)
            pred_mean = forecast_result.predicted_mean.iloc[0]
            conf_int = forecast_result.conf_int(alpha=0.05)  # 95% confidence interval
            lower_bound = conf_int.iloc[0, 0]  # Lower bound
            upper_bound = conf_int.iloc[0, 1]  # Upper bound

            predictions.append(pred_mean)
            lower_bounds.append(lower_bound)
            upper_bounds.append(upper_bound)

            # Update history with the new observation
            history_y = pd.concat([history_y, log_volume_test.iloc[t:t+1]])
            history_X = pd.concat([history_X, log_interest_test.iloc[t:t+1]])

        # Convert predictions to series
        predictions_series = pd.Series(predictions, index=test_index[:len(predictions)])
        lower_bounds_series = pd.Series(lower_bounds, index=test_index[:len(lower_bounds)])
        upper_bounds_series = pd.Series(upper_bounds, index=test_index[:len(upper_bounds)])
        actual_test_aligned = log_volume_test.loc[test_index[:len(predictions)]]

        # Drop any NaNs
        combined = pd.concat([actual_test_aligned, predictions_series, lower_bounds_series, upper_bounds_series], axis=1).dropna()
        actual_test_aligned = combined.iloc[:, 0]
        predictions_series = combined.iloc[:, 1]
        lower_bounds_series = combined.iloc[:, 2]
        upper_bounds_series = combined.iloc[:, 3]

        # Calculate Mean Squared Error
        rmse = mean_squared_error(actual_test_aligned, predictions_series, squared = False)
        print(f"Root mean Squared Error (RMSE) on the test set for ARMAX model: {rmse}")

        # Calculate correct rise/fall predictions
        actual_changes = actual_test_aligned.diff().dropna()
        predicted_changes = predictions_series.diff().dropna()

        correct_predictions = (np.sign(actual_changes) == np.sign(predicted_changes)).sum()
        total_predictions = len(predicted_changes)
        accuracy = correct_predictions / total_predictions * 100

        print(f"Correct Rise/Fall Predictions for ARMAX model: {correct_predictions}/{total_predictions}")
        print(f"Accuracy of predicting rise/fall for ARMAX model: {accuracy:.2f}%")
    else:
        print("No suitable ARMAX model was found during grid search.")

    # ================== ARMA Model ==================

    # Create combinations of p and q
    pdq = list(itertools.product(p, q))

    # Initialize variables to store the best ARMA model
    best_aic_arma = np.inf
    best_order_arma = None
    best_result_arma = None

    # Grid search over p and q for ARMA model
    print("\nStarting grid search over p and q for ARMA model allowing up to two insignificant parameters...")
    for order in pdq:
        print(f"Trying AR={order[0]}, MA={order[1]}")
        result_arma, aic_arma = fit_arma_model(log_volume_train, order[0], order[1])
        if result_arma is not None and aic_arma < best_aic_arma:
            best_aic_arma = aic_arma
            best_order_arma = order
            best_result_arma = result_arma

    # Check if a best ARMA model was found
    if best_result_arma is not None:
        print(f"\nBest ARMA model found: AR={best_order_arma[0]}, MA={best_order_arma[1]} with AIC={best_aic_arma}")
        print(best_result_arma.summary())

        # Perform residual analysis
        residuals_arma = best_result_arma.resid
        perform_jarque_bera_test(residuals_arma, "Best ARMA Model")

        # Rolling forecast with confidence intervals for ARMA model
        predictions_arma = []
        lower_bounds_arma = []  # Store lower bounds of confidence intervals
        upper_bounds_arma = []  # Store upper bounds of confidence intervals
        history_y_arma = log_volume_train.copy()
        test_index_arma = log_volume_test.index

        for t in range(len(log_volume_test)):
            # Fit the model on the history data
            y_aligned_arma = history_y_arma.dropna()
            model_arma = sm.tsa.SARIMAX(y_aligned_arma, order=(best_order_arma[0], 0, best_order_arma[1]), trend='c')
            result_arma = model_arma.filter(best_result_arma.params)

            # Forecast one step ahead, including confidence intervals
            forecast_result_arma = result_arma.get_forecast(steps=1)
            pred_mean_arma = forecast_result_arma.predicted_mean.iloc[0]
            conf_int_arma = forecast_result_arma.conf_int(alpha=0.05)  # 95% confidence interval
            lower_bound_arma = conf_int_arma.iloc[0, 0]  # Lower bound
            upper_bound_arma = conf_int_arma.iloc[0, 1]  # Upper bound

            predictions_arma.append(pred_mean_arma)
            lower_bounds_arma.append(lower_bound_arma)
            upper_bounds_arma.append(upper_bound_arma)

            # Update history with the new observation
            history_y_arma = pd.concat([history_y_arma, log_volume_test.iloc[t:t+1]])

        # Convert predictions to series
        predictions_series_arma = pd.Series(predictions_arma, index=test_index_arma[:len(predictions_arma)])
        lower_bounds_series_arma = pd.Series(lower_bounds_arma, index=test_index_arma[:len(lower_bounds_arma)])
        upper_bounds_series_arma = pd.Series(upper_bounds_arma, index=test_index_arma[:len(upper_bounds_arma)])
        actual_test_aligned_arma = log_volume_test.loc[test_index_arma[:len(predictions_arma)]]

        # Drop any NaNs
        combined_arma = pd.concat([actual_test_aligned_arma, predictions_series_arma, lower_bounds_series_arma, upper_bounds_series_arma], axis=1).dropna()
        actual_test_aligned_arma = combined_arma.iloc[:, 0]
        predictions_series_arma = combined_arma.iloc[:, 1]
        lower_bounds_series_arma = combined_arma.iloc[:, 2]
        upper_bounds_series_arma = combined_arma.iloc[:, 3]

        # Calculate Mean Squared Error
        rmse_arma = mean_squared_error(actual_test_aligned_arma, predictions_series_arma, squared = False)

        print(f"Root mean Squared Error (RMSE) on the test set for ARMA model: {rmse_arma}")
        print(f"The ARMAX model is {rmse_arma/rmse} times better than the ARMA in terms of RMSE")

        # Calculate correct rise/fall predictions
        actual_changes_arma = actual_test_aligned_arma.diff().dropna()
        predicted_changes_arma = predictions_series_arma.diff().dropna()

        correct_predictions_arma = (np.sign(actual_changes_arma) == np.sign(predicted_changes_arma)).sum()
        total_predictions_arma = len(predicted_changes_arma)
        accuracy_arma = correct_predictions_arma / total_predictions_arma * 100

        print(f"Correct Rise/Fall Predictions for ARMA model: {correct_predictions_arma}/{total_predictions_arma}")
        print(f"Accuracy of predicting rise/fall for ARMA model: {accuracy_arma:.2f}%")
    else:
        print("No suitable ARMA model was found during grid search.")

    # Combined Plotting Section
    if best_result is not None and best_result_arma is not None:
        # Combine training and test data for actual volume
        actual_volume_full = pd.concat([log_volume_train, log_volume_test])

        # Plot actual volume over the full date range
        plt.figure(figsize=(12, 6))
        plt.plot(log_volume_test.index, log_volume_test, label='Actual Volume', color='blue')

        # Plot ARMAX predictions
        plt.plot(predictions_series.index, predictions_series, label='ARMAX Predicted Volume', linestyle='--', color='green')
        plt.fill_between(predictions_series.index, lower_bounds_series, upper_bounds_series, color='green', alpha=0.2, label='ARMAX 95% Confidence Interval')

        # Plot ARMA predictions
        plt.plot(predictions_series_arma.index, predictions_series_arma, label='ARMA Predicted Volume', linestyle='-.', color='red')
        plt.fill_between(predictions_series_arma.index, lower_bounds_series_arma, upper_bounds_series_arma, color='red', alpha=0.2, label='ARMA 95% Confidence Interval')

        plt.title(f"Rolling Forecast: ARMAX and ARMA Predictions VS Actual Traded Volume for {symbol}")
        plt.xlabel('Date')
        plt.ylabel('Volume')
        plt.legend()
        plt.show()
        plt.close()
    else:
        print("Could not plot predictions as one or both models were not successfully fitted.")
0