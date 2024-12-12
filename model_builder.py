# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.stattools import jarque_bera
import warnings
import itertools
from sklearn.metrics import mean_squared_error
from getStockData import fetch_stock_data
from getTrendsData import fetch_google_trends_data
from scipy import stats
from scipy.stats import zscore

warnings.filterwarnings("ignore")

symbol = 'BTC-USD'
start_date1 = '2023-03-23'
end_date1 = '2023-11-23'
start_date2 = '2023-11-17'
end_date2 = '2024-07-17'

keyword = 'bitcoin'
folder = './'

# Define the ranges for p, q, and x_order (ALL SHOULD BE 7!)
p = range(0, 2)  # AR order 
q = range(0, 2)  # MA order
x_order_range = range(1, 2)  # Exogenous variable lags

def split_data(target, exog, train_size=0.8):

    # Combine and align indexes
    combined_index = target.index.union(exog.index)
    combined_index = combined_index.sort_values()  # Ensure the index is sorted

    target_aligned = target.reindex(combined_index).fillna(method='ffill')
    exog_aligned = exog.reindex(combined_index).fillna(method='ffill')

    # Compute the split point
    train_len = int(len(target_aligned) * train_size)

    # Split into training and testing sets
    target_train, target_test = target_aligned.iloc[:train_len], target_aligned.iloc[train_len:]
    exog_train, exog_test = exog_aligned.iloc[:train_len], exog_aligned.iloc[train_len:]

    return target_train, target_test, exog_train, exog_test, train_len


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
        # Return the exogenous variable column names
        exog_columns = X_lags.columns.tolist()
        # Check parameter significance
        p_values = result.pvalues
        # Allow models with at most two insignificant parameters (p-value >= 0.05)
        num_insignificant = (p_values >= 0.05).sum()
        if num_insignificant > 2:
            return None, None
    except Exception as e:
        # If the model fails to converge, return None for result
        print(f"Failed to fit model with AR={ar_order}, MA={ma_order}, X_lags={x_order}: {e}")
        result = None
        exog_columns = None

    return result, exog_columns

def prepare_exog_lags(X, x_order):
    """Prepare lagged exogenous variables up to x_order."""
    X_lags = pd.concat([X.shift(i) for i in range(1, x_order + 1)], axis=1)
    X_lags.columns = [f'X_lag_{i}' for i in range(1, x_order + 1)]
    return X_lags

# Define fit_arma_model function
def fit_arma_model(y, ar_order, ma_order):
    """Fit ARMA model using SARIMAX without exogenous variables."""
    y_aligned = y.dropna()
    # Fit the SARIMAX model with the AR and MA terms
    try:
        model_arma = sm.tsa.SARIMAX(y_aligned, order=(ar_order, 0, ma_order), trend='c')
        result = model_arma.fit(disp=False)
        # Check parameter significance
        p_values = result.pvalues
        # Allow models with at most two insignificant parameters (p-value >= 0.05)
        num_insignificant = (p_values >= 0.05).sum()
        if num_insignificant > 2:
            return None
    except Exception as e:
        # If the model fails to converge, return None for result
        result = None

    return result

def reverse_log_returns(log_returns, initial_value):
    """Reconstruct the original series from log returns."""
    log_returns = log_returns.astype(float)
    # Calculate cumulative sum of log returns
    cumulative_log_returns = np.cumsum(log_returns)
    # Reconstruct the original series
    reconstructed_series = initial_value * (10 ** cumulative_log_returns)
    return reconstructed_series


def detect_and_handle_outliers(series, threshold=3):
    """
    Detect and handle outliers in a time series using Z-score method.
    Points with Z-score greater than the threshold will be considered as outliers.
    Returns the cleaned series.
    """

    # Compute Z-scores
    z_scores = zscore(series)

    # Identify outliers where the Z-score is greater than the threshold
    outliers = np.abs(z_scores) > threshold

    # Handle outliers (set outliers to NaN, then fill them with linear interpolation)
    series[outliers] = np.nan
    series = series.interpolate()  # Interpolate NaN values

    return series

if __name__ == '__main__':

     # Perform ADF test
    #result = adfuller(log_volume_series_cleaned)
    #print(f"p-value: {result[1]}")
    #if result[1] < 0.05:
        #print("The Volume series is stationary (reject the null hypothesis).")
    #else:
        #print("The Volume series is not stationary (fail to reject the null hypothesis).")

# Fetch stock and trends data
    original_volume, trading_dates = fetch_stock_data(symbol, start_date1, end_date2)
    original_trends = fetch_google_trends_data(keyword, folder, trading_dates, start_date1, end_date1, start_date2, end_date2, plot=True)

# EDITED

# Log-transform volume series

    log_volume = np.log10(original_volume /original_volume.shift(1))

# Log-transform trend series 
    log_trends = np.log10(original_trends/original_trends.shift(1))

# Detect and handle outliers
    log_volume = detect_and_handle_outliers(log_volume)
    log_trends = detect_and_handle_outliers(log_trends)

    train_volume, test_volume, train_trends, test_trends, train_len = split_data(log_volume, log_trends)
        
    # Create combinations of p, q, and x_order
    pdq = list(itertools.product(p, q))
    x_orders = list(x_order_range)


    # ================== ARMAX Model ==================

    # Generate all (p, q) combinations for the AR and MA orders
    pq_combinations = list(itertools.product(p, q))

    # Initialize variables to store the best ARMAX model
    best_armax_aicc = np.inf
    best_armax_order = None
    best_armax_x_lag = None
    best_armax_result = None
    best_armax_exog_cols = None

    # Grid search over p, q, and x_lag based on AIC
    for ar_order, ma_order in pq_combinations:
        for x_lag in x_orders:
            current_result, current_exog_cols = fit_armax_model(train_volume, train_trends, ar_order, ma_order, x_lag)
            if current_result is not None:
                current_aicc = current_result.aicc
                if current_aicc < best_armax_aicc:
                    best_armax_aicc = current_aicc
                    best_armax_order = (ar_order, ma_order)
                    best_armax_x_lag = x_lag
                    best_armax_result = current_result
                    best_armax_exog_cols = current_exog_cols

    # Proceed only if we found a suitable ARMAX model
    if best_armax_result is not None:
        ar_order, ma_order = best_armax_order
        print(f"\nBest ARMAX model found: AR={ar_order}, MA={ma_order}, X_lags={best_armax_x_lag} with AIC={best_armax_aicc}")
        print(best_armax_result.summary())

        # Rolling forecast (one-step-ahead) with confidence intervals
        armax_predictions = []
        armax_lower_bounds = []
        armax_upper_bounds = []
        history_y_armax = train_volume.copy()
        history_x_armax = train_trends.copy()
        test_idx_armax = test_volume.index

        for t in range(len(test_volume)):
            # Prepare lagged exogenous variables
            exog_lags = pd.concat([history_x_armax.shift(i) for i in range(1, best_armax_x_lag + 1)], axis=1)
            exog_lags.columns = [f'X_lag_{i}' for i in range(1, best_armax_x_lag + 1)]
            exog_lags = sm.add_constant(exog_lags).dropna()

            # Align endogenous data
            y_aligned_armax = history_y_armax.loc[exog_lags.index]

            # Update model with current data
            model_armax = sm.tsa.SARIMAX(y_aligned_armax, exog=exog_lags, order=(ar_order, 0, ma_order))
            updated_armax_result = model_armax.filter(best_armax_result.params)

            # Prepare exogenous variables for forecasting
            exog_forecast_lags = []
            for lag in range(1, best_armax_x_lag + 1):
                idx = -lag + t + 1
                # If idx >= 0, we are in the test set; otherwise, in the training extension
                exog_value = test_trends.iloc[idx] if idx >= 0 else history_x_armax.iloc[idx]
                exog_forecast_lags.append(exog_value)

            # Reverse order to match lag naming (X_lag_1 is most recent)
            exog_forecast_lags = exog_forecast_lags[::-1]
            exog_forecast_values = [1] + exog_forecast_lags
            exog_forecast_df = pd.DataFrame([exog_forecast_values], columns=best_armax_exog_cols)

            # Forecast one step ahead
            forecast_result_armax = updated_armax_result.get_forecast(steps=1, exog=exog_forecast_df)
            pred_mean_armax = forecast_result_armax.predicted_mean.iloc[0]
            conf_int_armax = forecast_result_armax.conf_int(alpha=0.05)  # 95% CI

            armax_predictions.append(pred_mean_armax)
            armax_lower_bounds.append(conf_int_armax.iloc[0, 0])
            armax_upper_bounds.append(conf_int_armax.iloc[0, 1])

            # Update history with the new actual observation
            history_y_armax = pd.concat([history_y_armax, test_volume.iloc[t:t+1]])
            history_x_armax = pd.concat([history_x_armax, test_trends.iloc[t:t+1]])

        # Align predictions and actual values
        armax_predictions_series = pd.Series(armax_predictions, index=test_idx_armax[:len(armax_predictions)])
        armax_lower_series = pd.Series(armax_lower_bounds, index=test_idx_armax[:len(armax_lower_bounds)])
        armax_upper_series = pd.Series(armax_upper_bounds, index=test_idx_armax[:len(armax_upper_bounds)])
        actual_test_armax = test_volume.loc[test_idx_armax[:len(armax_predictions)]]

        # Drop NaNs and align data
        armax_combined = pd.concat([actual_test_armax, armax_predictions_series, armax_lower_series, armax_upper_series], axis=1).dropna()
        actual_test_armax = armax_combined.iloc[:, 0]
        armax_predictions_series = armax_combined.iloc[:, 1]

        # Calculate rise/fall prediction accuracy
        actual_changes_armax = actual_test_armax.diff().dropna()
        predicted_changes_armax = armax_predictions_series.diff().dropna()
        correct_armax = (np.sign(actual_changes_armax) == np.sign(predicted_changes_armax)).sum()
        total_armax = len(predicted_changes_armax)
        accuracy_armax = (correct_armax / total_armax) * 100

        print(f"Correct Rise/Fall Predictions for ARMAX model: {correct_armax}/{total_armax}")
        print(f"Accuracy of predicting rise/fall for ARMAX model: {accuracy_armax:.2f}%")

    else:
        print("No suitable ARMAX model was found during grid search.")


    # ================== ARMA Model ==================

    # Generate all (p, q) combinations
    pq_combinations = list(itertools.product(p, q))

    # Initialize variables for the best ARMA model
    best_arma_aicc = np.inf
    best_arma_order = None
    best_arma_result = None

    # Grid search over (p, q) to find the best ARMA model based on AIC
    for ar_order, ma_order in pq_combinations:
        current_result = fit_arma_model(train_volume, ar_order, ma_order)
        if current_result is not None:
            current_aicc = current_result.aicc
            if current_aicc < best_arma_aicc:
                best_arma_aicc = current_aicc
                best_arma_order = (ar_order, ma_order)
                best_arma_result = current_result

    # Proceed only if we found a suitable ARMA model
    if best_arma_result is not None:
        ar_order, ma_order = best_arma_order
        print(f"\nBest ARMA model found: AR={ar_order}, MA={ma_order} with AIC={best_arma_aicc}")
        print(best_arma_result.summary())

        # Rolling forecast with confidence intervals
        # Note: This is a one-step-ahead forecast loop over the test period.

        arma_predictions = []
        arma_lower_bounds = []
        arma_upper_bounds = []
        history_y_arma = train_volume.copy()
        test_idx_arma = test_volume.index

        # Perform one-step-ahead forecasts over the test set
        for t in range(len(test_volume)):
            y_fit_data = history_y_arma.dropna()
            # Re-estimate the model at each step with the best parameters
            model_arma = sm.tsa.SARIMAX(y_fit_data, order=(ar_order, 0, ma_order), trend='c')
            updated_result_arma = model_arma.filter(best_arma_result.params)

            # Forecast one step ahead
            forecast_result_arma = updated_result_arma.get_forecast(steps=1)
            pred_mean_arma = forecast_result_arma.predicted_mean.iloc[0]
            conf_int_arma = forecast_result_arma.conf_int(alpha=0.05)  # 95% CI

            arma_predictions.append(pred_mean_arma)
            arma_lower_bounds.append(conf_int_arma.iloc[0, 0])
            arma_upper_bounds.append(conf_int_arma.iloc[0, 1])

            # Update history with the new actual value
            history_y_arma = pd.concat([history_y_arma, test_volume.iloc[t:t+1]])

        # Align predictions and actual values for evaluation
        arma_predictions_series = pd.Series(arma_predictions, index=test_idx_arma[:len(arma_predictions)])
        arma_lower_series = pd.Series(arma_lower_bounds, index=test_idx_arma[:len(arma_lower_bounds)])
        arma_upper_series = pd.Series(arma_upper_bounds, index=test_idx_arma[:len(arma_upper_bounds)])
        actual_test_arma = test_volume.loc[test_idx_arma[:len(arma_predictions)]]

        # Combine into a single DataFrame and drop NaNs for evaluation
        arma_combined = pd.concat([actual_test_arma, arma_predictions_series, arma_lower_series, arma_upper_series], axis=1).dropna()
        actual_test_arma = arma_combined.iloc[:, 0]
        arma_predictions_series = arma_combined.iloc[:, 1]

        # Evaluate forecasts using RMSE
        rmse_arma_test = mean_squared_error(actual_test_arma, arma_predictions_series, squared=False)
        rmse_armax_test = mean_squared_error(actual_test_armax, armax_predictions_series, squared=False) 

        print(f"RMSE (ARMAX): {rmse_armax_test}")
        print(f"RMSE (ARMA): {rmse_arma_test}")
        print(f"The ARMAX model is {rmse_arma_test / rmse_armax_test:.2f} times better than the ARMA model based on RMSE")

        # Evaluate rise/fall accuracy
        actual_changes_arma = actual_test_arma.diff().dropna()
        predicted_changes_arma = arma_predictions_series.diff().dropna()
        correct_arma = (np.sign(actual_changes_arma) == np.sign(predicted_changes_arma)).sum()
        total_arma = len(predicted_changes_arma)
        accuracy_arma = (correct_arma / total_arma) * 100

        print(f"Correct Rise/Fall Predictions for ARMA model: {correct_arma}/{total_arma}")
        print(f"Accuracy (rise/fall) for ARMA model: {accuracy_arma:.2f}%")

    else:
        print("No suitable ARMA model found during grid search.")

        # Combined Plotting Section
    if best_armax_result is not None and best_arma_result is not None:
        # Combine training and test data for actual volume (in log returns form)
        actual_volume_full = pd.concat([train_volume, test_volume])

        last_train_date = train_volume.index[-1]
        initial_value = original_volume[last_train_date]

        # Reconstruct actual volumes from log returns
        actual_test_reconstructed = reverse_log_returns(test_volume, initial_value)
        reconstructed_predictions_armax = reverse_log_returns(armax_predictions_series, initial_value)
        reconstructed_predictions_arma = reverse_log_returns(arma_predictions_series, initial_value)

        # Keep commented code as is (not removing or modifying):
        #lower_bounds_reconstructed_armax = reverse_log_returns(lower_bounds_series, initial_value_test)
        #upper_bounds_reconstructed_armax = reverse_log_returns(upper_bounds_series, initial_value_test)

        #lower_bounds_reconstructed_arma = reverse_log_returns(lower_bounds_series_arma, initial_value_test)
        #upper_bounds_reconstructed_arma = reverse_log_returns(upper_bounds_series_arma, initial_value_test)

        plt.figure(figsize=(12, 6))

        # Plot the reconstructed actual volumes (blue)
        plt.plot(actual_test_reconstructed.index, actual_test_reconstructed, label='Test volume (outliers handled)', color='blue')

        actual_volume_test = original_volume[train_len:]

        # Plot the original test_volume (black)
        plt.plot(actual_volume_test.index, actual_volume_test, label='Test Volume (outliers not handled) ', color='black')

        # Plot ARMAX predictions (reconstructed to actual volumes)
        plt.plot(reconstructed_predictions_armax.index, reconstructed_predictions_armax,
                label='ARMAX Predicted Volume', linestyle='--', color='green')
        #plt.fill_between(predictions_reconstructed_armax.index, lower_bounds_series, upper_bounds_series, color='green', alpha=0.2, label='ARMAX 95% Confidence Interval')

        # Plot ARMA predictions (reconstructed to actual volumes)
        plt.plot(reconstructed_predictions_arma.index, reconstructed_predictions_arma,
                label='ARMA Predicted Volume', linestyle='-.', color='red')
        #plt.fill_between(predictions_reconstructed_arma.index, lower_bounds_series_arma, upper_bounds_series_arma, color='red', alpha=0.2, label='ARMA 95% Confidence Interval')

        plt.legend(loc='lower left', fontsize='8')
        plt.title(f"Rolling Forecast: ARMAX and ARMA Predictions vs. Actual Traded Volume for {symbol}")
        plt.xlabel('Date')
        plt.ylabel('Volume')
        plt.show()
        plt.close()

    else:
        print("Could not plot predictions as one or both models were not successfully fitted.")
