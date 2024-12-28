import matplotlib.pyplot as plt
import numpy as np

# Categories (in the table order)
categories = ['bitcoin', 'ethereum', 'tether', 'cardano', 'polkadot', 'vechain']

# ----- Data from the updated table -----

# Rise and Fall percentages (store them numerically, even though they have '%' in the table)
arma_rise_fall =  [78.12, 75.00, 77.08, 73.96, 73.96, 73.96]
armax_rise_fall = [77.08, 78.12, 75.00, 69.79, 75.00, 71.88]

# RMSE values
arma_rmse =  [0.302, 0.299, 0.267, 0.336, 0.352, 0.300]
armax_rmse = [0.297, 0.287, 0.273, 0.365, 0.334, 0.301]

# Set up x positions
x = np.arange(len(categories))
bar_width = 0.35

# ----- 1) Bar chart for Rise and fall -----
plt.figure(figsize=(8, 5))
plt.bar(x - bar_width/2, arma_rise_fall,  width=bar_width, label='ARMA',  color='blue')
plt.bar(x + bar_width/2, armax_rise_fall, width=bar_width, label='ARMAX', color='orange')

plt.xticks(x, categories, rotation=20)  # rotate if needed
plt.ylabel('Rise and Fall (%)')
plt.title('Rise and Fall Comparison (ARMA vs ARMAX)')
plt.legend()
plt.tight_layout()
plt.show()

# ----- 2) Bar chart for RMSE -----
plt.figure(figsize=(8, 5))
plt.bar(x - bar_width/2, arma_rmse,  width=bar_width, label='ARMA',  color='blue')
plt.bar(x + bar_width/2, armax_rmse, width=bar_width, label='ARMAX', color='orange')

plt.xticks(x, categories, rotation=20)
plt.ylabel('RMSE')
plt.title('RMSE Comparison (ARMA vs ARMAX)')
plt.legend()
plt.tight_layout()
plt.show()
