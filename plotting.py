import matplotlib.pyplot as plt
import numpy as np

# Categories
categories = ['bitcoin', 'ethereum', 'tether', 'cardano', 'polkadot', 'vechain']

# Rise and Fall data
arma_rise_fall = [78.12, 75.00, 77.08, 73.96, 73.96, 73.96]
armax_rise_fall = [77.08, 78.12, 75.00, 69.79, 75.00, 71.88]

# Confidence intervals for Rise and Fall
arma_rise_fall_ci = [(69.79, 85.44), (69.79, 85.44), (68.75, 84.40), (64.58, 82.29), (65.62, 83.33), (64.58, 82.29)]
armax_rise_fall_ci = [(68.75, 85.42), (68.75, 85.42), (66.67, 83.33), (60.42, 78.12), (65.62, 82.32), (62.50, 80.21)]

# RMSE data
arma_rmse = [0.302, 0.299, 0.267, 0.336, 0.352, 0.300]
armax_rmse = [0.297, 0.287, 0.273, 0.365, 0.334, 0.301]

# Confidence intervals for RMSE
arma_rmse_ci = [(0.2408, 0.3178), (0.2327, 0.3270), (0.1967, 0.2733), (0.2695, 0.3457), (0.2578, 0.3611), (0.2405, 0.3153)]
armax_rmse_ci = [(0.2380, 0.3170), (0.2380, 0.3170), (0.2020, 0.2720), (0.2941, 0.3838), (0.2378, 0.3361), (0.2407, 0.3162)]

# Set up x positions
x = np.arange(len(categories))
bar_width = 0.35

# Function to draw bars with error hats
def add_error_bars(bar_positions, data, confidence_intervals, color):
    for i, pos in enumerate(bar_positions):
        lower = confidence_intervals[i][0]
        upper = confidence_intervals[i][1]
        mid = data[i]
        # Vertical line
        plt.plot([pos, pos], [lower, upper], color=color, linestyle='-', linewidth=1.5)
        # Horizontal hats
        cap_width = 0.1
        plt.plot([pos - cap_width, pos + cap_width], [lower, lower], color=color, linestyle='-', linewidth=1.5)
        plt.plot([pos - cap_width, pos + cap_width], [upper, upper], color=color, linestyle='-', linewidth=1.5)

# Plot for Rise and Fall
plt.figure(figsize=(8, 5))
arma_bars = plt.bar(x - bar_width/2, arma_rise_fall, width=bar_width, color='blue', label='ARMA')
armax_bars = plt.bar(x + bar_width/2, armax_rise_fall, width=bar_width, color='orange', label='ARMAX')

# Add error bars with hats for Rise and Fall
add_error_bars(x - bar_width/2, arma_rise_fall, arma_rise_fall_ci, color='black')
add_error_bars(x + bar_width/2, armax_rise_fall, armax_rise_fall_ci, color='black')

plt.xticks(x, categories)
plt.ylabel('Rise and Fall (%)')
plt.title('Rise and Fall Comparison')
plt.legend()
plt.tight_layout()
plt.show()

# Plot for RMSE
plt.figure(figsize=(8, 5))
arma_bars = plt.bar(x - bar_width/2, arma_rmse, width=bar_width, color='blue', label='ARMA')
armax_bars = plt.bar(x + bar_width/2, armax_rmse, width=bar_width, color='orange', label='ARMAX')

# Add error bars with hats for RMSE
add_error_bars(x - bar_width/2, arma_rmse, arma_rmse_ci, color='black')
add_error_bars(x + bar_width/2, armax_rmse, armax_rmse_ci, color='black')

plt.xticks(x, categories)
plt.ylabel('RMSE')
plt.title('RMSE Comparison')
plt.legend()
plt.tight_layout()
plt.show()
