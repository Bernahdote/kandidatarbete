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

warnings.filterwarnings("ignore")

symbol = 'BTC-USD'
start_date1 = '2023-03-23'
end_date1 = '2023-11-23'
start_date2 = '2023-11-17'
end_date2 = '2024-07-17'

keyword = 'bitcoin'
folder = './'

# Define the ranges for p, q, and x_order
p = range(0, 7)  # AR order
q = range(0, 7)  # MA order
x_order_range = range(1, 7)  # Exogenous variable lags

def split_data(target, features, train_size=0.8):
    """Splits target and features into aligned train and test sets over the union of their indexes."""
    # Align both target and features over the union of their indexes
    combined_index = target.index.union(features.index)
    target_aligned = target.reindex(combined_index)
    features_aligned = features.reindex(combined_index)

    # Fill missing values (forward-fill or interpolate)
    target_aligned = target_aligned.fillna(method='ffill')
    features_aligned = features_aligned.fillna(method='ffill')

    # Calculate the number of training samples
    train_sample_count = int(len(target_aligned) * train_size)

    # Split the aligned data into training and testing sets
    target_train, target_test = target_aligned.iloc[:train_sample_count], target_aligned.iloc[train_sample_count:]
    features_train, features_test = features_aligned.iloc[:train_sample_count], features_aligned.iloc[train_sample_count:]

    return target_train, target_test, features_train, features_test

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
        print(f"Failed to fit model with AR={ar_order}, MA={ma_order}, X_lags={x_order}: {e}")
        result = None
        exog_columns = None

    return result, exog_columns

def prepare_exog_lags(X, x_order):
    """Prepare lagged exogenous variables up to x_order."""
    X_lags = pd.concat([X.shift(i) for i in range(1, x_order + 1)], axis=1)
    X_lags.columns = [f'X_lag_{i}' for i in range(1, x_order + 1)]
    return X_lags

def fit_arma_model(y, ar_order, ma_order):
    """Fit ARMA model using SARIMAX without exogenous variables."""
    y_aligned = y.dropna()
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
        print(f"Failed to fit ARMA model with AR={ar_order}, MA={ma_order}: {e}")
        return None

    return result

if __name__ == '__main__':

    # ===========================
    # 1) Fetch stock and trends data
    # ===========================
    # log_volume_series: log of trading volume
    # trading_dates: list or index of trading dates
    log_volume_series, trading_dates = fetch_stock_data(symbol, start_date1, end_date2)
    log_interest_series = fetch_google_trends_data(keyword, folder, trading_dates,
                                                   start_date1, end_date1, start_date2, end_date2, plot=True)

    plt.figure(figsize=(12, 6))

    # ===========================
    # 2) Split data (train/test)
    # ===========================
    log_volume_train, log_volume_test, log_interest_train, log_interest_test = split_data(
        log_volume_series, log_interest_series, train_size=0.8
    )

    # Create combinations of p, q, and x_order
    pdq = list(itertools.product(p, q))
    x_orders = list(x_order_range)

    # ===========================
    # 3) Grid Search for ARMAX
    # ===========================
    best_aicc_armax = np.inf
    best_order = None
    best_x_order = None
    best_result_armax = None
    best_exog_columns = None

    for order in pdq:
        for x_order in x_orders:
            result, exog_columns = fit_armax_model(log_volume_train, log_interest_train,
                                                   ar_order=order[0], ma_order=order[1], x_order=x_order)
            if result is not None:
                aicc_armax = result.aicc
                if aicc_armax < best_aicc_armax:
                    best_aicc_armax = aicc_armax
                    best_order = order
                    best_x_order = x_order
                    best_result_armax = result
                    best_exog_columns = exog_columns

    # ===========================
    # 4) Evaluate Best ARMAX
    # ===========================
    if best_result_armax is not None:
        print(f"\nBest ARMAX model found: AR={best_order[0]}, MA={best_order[1]}, X_lags={best_x_order} "
              f"with AICc={best_aicc_armax}")
        print(best_result_armax.summary())

        # Rolling forecast with confidence intervals
        predictions_armax = []
        lower_bounds_armax = []
        upper_bounds_armax = []
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

            # Re-fit the model on the history data, using the best_result_armax params as starting values
            model = sm.tsa.SARIMAX(y_aligned, exog=X_lags, order=(best_order[0], 0, best_order[1]))
            result = model.filter(best_result_armax.params)

            # Prepare exogenous variables for forecasting (single-step forecast)
            exog_forecast_lags = []
            for lag in range(1, best_x_order + 1):
                idx = -lag + t + 1
                if idx >= 0:
                    exog_value = log_interest_test.iloc[idx]
                else:
                    exog_value = history_X.iloc[idx]
                exog_forecast_lags.append(exog_value)
            # Reverse to match the lag order (if needed)
            exog_forecast_lags = exog_forecast_lags[::-1]

            # Combine with constant term
            exog_forecast_values = [1] + exog_forecast_lags
            exog_forecast_df = pd.DataFrame([exog_forecast_values], columns=best_exog_columns)

            # Forecast one step ahead, including confidence intervals
            forecast_result = result.get_forecast(steps=1, exog=exog_forecast_df)
            pred_mean = forecast_result.predicted_mean.iloc[0]
            conf_int = forecast_result.conf_int(alpha=0.05)  # 95% confidence interval
            lower_armax = conf_int.iloc[0, 0]  # Lower bound
            upper_armax = conf_int.iloc[0, 1]  # Upper bound

            predictions_armax.append(pred_mean)
            lower_bounds_armax.append(lower_armax)
            upper_bounds_armax.append(upper_armax)

            # Update history with the new observation
            history_y = pd.concat([history_y, log_volume_test.iloc[t:t+1]])
            history_X = pd.concat([history_X, log_interest_test.iloc[t:t+1]])

        # Convert predictions to a Series
        predictions_armax = pd.Series(predictions_armax, index=test_index[:len(predictions_armax)])
        lower_armax = pd.Series(lower_bounds_armax, index=test_index[:len(lower_bounds_armax)])
        upper_armax = pd.Series(upper_bounds_armax, index=test_index[:len(upper_bounds_armax)])

        # Align actual with predictions
        actual_test_aligned_armax = log_volume_test.loc[test_index[:len(predictions_armax)]]
        combined_armax = pd.concat([actual_test_aligned_armax, predictions_armax,
                                    lower_armax, upper_armax], axis=1).dropna()
        actual_test_aligned_armax = combined_armax.iloc[:, 0]
        predictions_armax = combined_armax.iloc[:, 1]
        lower_armax = combined_armax.iloc[:, 2]
        upper_armax = combined_armax.iloc[:, 3]

        # ===========================
        # 4a) Rise/Fall accuracy (LOG scale)
        # ===========================
        actual_changes_armax = actual_test_aligned_armax.diff().dropna()
        predicted_changes_armax = predictions_armax.diff().dropna()

        correct_predictions_armax = (np.sign(actual_changes_armax) == np.sign(predicted_changes_armax)).sum()
        total_predictions_armax = len(predicted_changes_armax)
        accuracy_log_armax = correct_predictions_armax / total_predictions_armax * 100

        print(f"Correct Rise/Fall Predictions (log scale) for ARMAX: {correct_predictions_armax}/{total_predictions_armax}")
        print(f"Accuracy (log scale) for ARMAX: {accuracy_log_armax:.2f}%")

    else:
        print("No suitable ARMAX model was found during grid search.")
        predictions_armax = None
        actual_test_aligned_armax = None
        lower_armax = None
        upper_armax = None

    # ===========================
    # 5) Grid Search for ARMA
    # ===========================
    best_aicc_arma = np.inf
    best_order_arma = None
    best_result_arma = None

    for order in pdq:
        result_arma = fit_arma_model(log_volume_train, order[0], order[1])
        if result_arma is not None:
            aicc_arma = result_arma.aicc
            if aicc_arma < best_aicc_arma:
                best_aicc_arma = aicc_arma
                best_order_arma = order
                best_result_arma = result_arma

    # ===========================
    # 6) Evaluate Best ARMA
    # ===========================
    if best_result_arma is not None:
        print(f"\nBest ARMA model found: AR={best_order_arma[0]}, MA={best_order_arma[1]} with AICc={best_aicc_arma}")
        print(best_result_arma.summary())

        # Rolling forecast with confidence intervals for ARMA
        predictions_arma = []
        lower_bounds_arma = []
        upper_bounds_arma = []
        history_y_arma = log_volume_train.copy()
        test_index_arma = log_volume_test.index

        for t in range(len(log_volume_test)):
            # Fit the model on the history data, reusing best_result_arma params
            y_aligned_arma = history_y_arma.dropna()
            model_arma = sm.tsa.SARIMAX(y_aligned_arma, order=(best_order_arma[0], 0, best_order_arma[1]), trend='c')
            result_arma = model_arma.filter(best_result_arma.params)

            # Forecast one step ahead
            forecast_result_arma = result_arma.get_forecast(steps=1)
            pred_mean_arma = forecast_result_arma.predicted_mean.iloc[0]
            conf_int_arma = forecast_result_arma.conf_int(alpha=0.05)
            lower_bound_arma = conf_int_arma.iloc[0, 0]
            upper_bound_arma = conf_int_arma.iloc[0, 1]

            predictions_arma.append(pred_mean_arma)
            lower_bounds_arma.append(lower_bound_arma)
            upper_bounds_arma.append(upper_bound_arma)

            # Update history
            history_y_arma = pd.concat([history_y_arma, log_volume_test.iloc[t:t+1]])

        # Convert to Series
        predictions_arma = pd.Series(predictions_arma, index=test_index_arma[:len(predictions_arma)])
        lower_arma = pd.Series(lower_bounds_arma, index=test_index_arma[:len(lower_bounds_arma)])
        upper_arma = pd.Series(upper_bounds_arma, index=test_index_arma[:len(upper_bounds_arma)])

        # Align actual with predictions
        actual_test_aligned_arma = log_volume_test.loc[test_index_arma[:len(predictions_arma)]]
        combined_arma = pd.concat([actual_test_aligned_arma, predictions_arma,
                                   lower_arma, upper_arma], axis=1).dropna()
        actual_test_aligned_arma = combined_arma.iloc[:, 0]
        predictions_arma = combined_arma.iloc[:, 1]
        lower_arma = combined_arma.iloc[:, 2]
        upper_arma = combined_arma.iloc[:, 3]

        # ===========================
        # 6a) Rise/Fall accuracy (LOG scale)
        # ===========================
        actual_changes_arma = actual_test_aligned_arma.diff().dropna()
        predicted_changes_arma = predictions_arma.diff().dropna()
        correct_predictions_arma = (np.sign(actual_changes_arma) == np.sign(predicted_changes_arma)).sum()
        total_predictions_arma = len(predicted_changes_arma)
        accuracy_log_arma = correct_predictions_arma / total_predictions_arma * 100

        print(f"Correct Rise/Fall Predictions (log scale) for ARMA: {correct_predictions_arma}/{total_predictions_arma}")
        print(f"Accuracy (log scale) for ARMA: {accuracy_log_arma:.2f}%")

    else:
        print("No suitable ARMA model was found during grid search.")
        predictions_arma = None
        actual_test_aligned_arma = None
        lower_arma = None
        upper_arma = None

    # ===========================
    # 7) RMSE Comparison in LOG scale (Optional)
    # ===========================
    if (best_result_armax is not None) and (best_result_arma is not None):
        rmse_armax_log = mean_squared_error(actual_test_aligned_armax, predictions_armax, squared=False)
        rmse_arma_log = mean_squared_error(actual_test_aligned_arma, predictions_arma, squared=False)

        print(f"\nRMSE (log scale) ARMAX: {rmse_armax_log}")
        print(f"RMSE (log scale) ARMA:  {rmse_arma_log}")
        ratio = (rmse_arma_log / rmse_armax_log) if rmse_armax_log != 0 else np.inf
        print(f"The ARMAX model is {ratio:.2f} times better than the ARMA model (log-scale RMSE)")
    else:
        print("Cannot compare RMSE in log scale since at least one model was not fitted.")

    # ===========================
    # 8) Exponentiate for Real-Scale RMSE & Accuracy
    # ===========================
    if (best_result_armax is not None) and (best_result_arma is not None):

        # ----- ARMAX -----
        actual_armax_exp = np.exp(actual_test_aligned_armax)
        pred_armax_exp   = np.exp(predictions_armax)
        lower_armax_exp  = np.exp(lower_armax)
        upper_armax_exp  = np.exp(upper_armax)

        # ----- ARMA -----
        actual_arma_exp = np.exp(actual_test_aligned_arma)
        pred_arma_exp   = np.exp(predictions_arma)
        lower_arma_exp  = np.exp(lower_arma)
        upper_arma_exp  = np.exp(upper_arma)

        # -- Real-scale RMSE --
        rmse_armax_real = mean_squared_error(actual_armax_exp, pred_armax_exp, squared=False)
        rmse_arma_real  = mean_squared_error(actual_arma_exp,  pred_arma_exp,  squared=False)
        print(f"\nRMSE (real scale) for ARMAX: {rmse_armax_real:.4f}")
        print(f"RMSE (real scale) for ARMA:  {rmse_arma_real:.4f}")
        
        # -- Real-scale Rise/Fall Accuracy --

        # ARMAX
        actual_armax_diff = actual_armax_exp.diff().dropna()
        pred_armax_diff   = pred_armax_exp.diff().dropna()
        # Align if needed
        combined_armax_diff = pd.concat([actual_armax_diff, pred_armax_diff], axis=1).dropna()
        combined_armax_diff.columns = ['actual_diff_armax', 'pred_diff_armax']
        correct_armax = (np.sign(combined_armax_diff['actual_diff_armax']) ==
                         np.sign(combined_armax_diff['pred_diff_armax'])).sum()
        total_armax = len(combined_armax_diff)
        armax_accuracy_real = correct_armax / total_armax * 100

        print(f"\nARMAX Rise/Fall Accuracy (real scale): {armax_accuracy_real:.2f}% "
              f"({correct_armax}/{total_armax})")

        # ARMA
        actual_arma_diff = actual_arma_exp.diff().dropna()
        pred_arma_diff   = pred_arma_exp.diff().dropna()
        combined_arma_diff = pd.concat([actual_arma_diff, pred_arma_diff], axis=1).dropna()
        combined_arma_diff.columns = ['actual_diff_arma', 'pred_diff_arma']
        correct_arma = (np.sign(combined_arma_diff['actual_diff_arma']) ==
                        np.sign(combined_arma_diff['pred_diff_arma'])).sum()
        total_arma = len(combined_arma_diff)

        # Plot actual volume over the full date range
        plt.figure(figsize=(12, 6))
        plt.plot(log_volume_test.index, np.exp(log_volume_test), label='Actual Volume', color='blue')

        # Plot ARMAX predictions
        plt.plot(pred_armax_exp.index, pred_armax_exp, label='ARMAX Predicted Volume', linestyle='--', color='green')
        plt.fill_between(pred_armax_exp.index, lower_armax_exp, upper_armax_exp, color='green', alpha=0.2, label='ARMAX 95% Confidence Interval')

        # Plot ARMA predictions
        plt.plot(pred_arma_exp.index, pred_arma_exp, label='ARMA Predicted Volume', linestyle='-.', color='red')
        plt.fill_between(pred_arma_exp.index, lower_arma_exp, upper_arma_exp, color='red', alpha=0.2, label='ARMA 95% Confidence Interval')

        plt.legend(loc='lower left', fontsize = '8')
        plt.title(f"Rolling Forecast: ARMAX and ARMA Predictions VS Actual Traded Volume for {symbol}")
        plt.xlabel('Date')
        plt.ylabel('Volume')
        plt.show()
        plt.close()
        

        def rmse_ci(actual: pd.Series, predicted: pd.Series, n_bootstraps=1000, alpha=0.05):
            """Bootstrap the RMSE and return the lower and upper confidence limits."""
            np.random.seed(42)
            # Align Series and drop missing
            combined = pd.concat([actual, predicted], axis=1).dropna()
            y_true = combined.iloc[:, 0].values
            y_pred = combined.iloc[:, 1].values
            n = len(combined)
    
            metric_values = []
            for _ in range(n_bootstraps):
                indices = np.random.randint(0, n, n)  # sample with replacement
                metric_values.append(np.sqrt(np.mean((y_true[indices] - y_pred[indices])**2)))
        
                lower = np.percentile(metric_values, alpha/2 * 100)
                upper = np.percentile(metric_values, (1 - alpha/2) * 100)
            return lower, upper


def sign_accuracy_ci(actual: pd.Series, predicted: pd.Series, n_bootstraps=1000, alpha=0.05):
    """
    Bootstrap the sign accuracy (rise/fall) and return the lower/upper confidence bounds.
    We'll consider sign(actual_t - actual_(t-1)) == sign(pred_t - pred_(t-1)).
    """
    np.random.seed(42)
    # Compute daily (or step) differences, drop missing
    actual_diff = actual.diff().dropna()
    pred_diff = predicted.diff().dropna()
    combined = pd.concat([actual_diff, pred_diff], axis=1).dropna()
    
    # 1 if sign is correct, 0 otherwise
    correctness = (np.sign(combined.iloc[:, 0]) == np.sign(combined.iloc[:, 1])).astype(int)
    n = len(correctness)
    
    metric_values = []
    for _ in range(n_bootstraps):
        indices = np.random.randint(0, n, n)
        metric_values.append(np.mean(correctness.iloc[indices]))
    
    # Convert mean correctness to percentage
    lower = np.percentile(metric_values, alpha/2 * 100) * 100
    upper = np.percentile(metric_values, (1 - alpha/2) * 100) * 100
    return lower, upper

# ================================
# Example usage after your final evaluation:
# ================================

# Suppose you already have these computed:
# actual_test_aligned_armax, predictions_armax
# actual_test_aligned_arma, predictions_arma

if (best_result_armax is not None) and (predictions_armax is not None):
    rmse_lower, rmse_upper = rmse_ci(actual_test_aligned_armax, predictions_armax)
    print(f"95% CI for ARMAX RMSE (log scale): [{rmse_lower:.4f}, {rmse_upper:.4f}]")

    sa_lower, sa_upper = sign_accuracy_ci(actual_test_aligned_armax, predictions_armax)
    print(f"95% CI for ARMAX Rise/Fall accuracy (log scale): [{sa_lower:.2f}%, {sa_upper:.2f}%]")

if (best_result_arma is not None) and (predictions_arma is not None):
    rmse_lower, rmse_upper = rmse_ci(actual_test_aligned_arma, predictions_arma)
    print(f"95% CI for ARMA RMSE (log scale): [{rmse_lower:.4f}, {rmse_upper:.4f}]")

    sa_lower, sa_upper = sign_accuracy_ci(actual_test_aligned_arma, predictions_arma)
    print(f"95% CI for ARMA Rise/Fall accuracy (log scale): [{sa_lower:.2f}%, {sa_upper:.2f}%]")