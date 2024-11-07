import matplotlib.pyplot as plt
import pandas as pd

# Data extracted from the image
data = {
    "Searchword": ["bitcoin", "bitcoin", "ethereum", "ethereum", "tether", "tether", "polkadot", "polkadot", "vechain", "vechain"],
    "Model": ["ARMA", "ARMAX", "ARMA", "ARMAX", "ARMA", "ARMAX", "ARMA", "ARMAX", "ARMA", "ARMAX"],
    "Rise and fall (%)": [81.25, 77.08, 72.92, 71.88, 78.12, 73.96, 65.62, 75.00, 71.88, 73.96],
    "RMSE": [0.273, 0.278, 0.277, 0.271, 0.244, 0.248, 0.297, 0.289, 0.275, 0.275]
}

# Create DataFrame
df = pd.DataFrame(data)

# Pivot the DataFrame to format it for grouped bar plotting
pivot_df_rise_and_fall = df.pivot(index="Searchword", columns="Model", values="Rise and fall (%)")
pivot_df_rmse = df.pivot(index="Searchword", columns="Model", values="RMSE")

# Plotting Rise and Fall Percentage
plt.figure(figsize=(10, 6))
pivot_df_rise_and_fall.plot(kind="bar", edgecolor="black")
plt.ylabel("Rise and Fall (%)")
plt.xlabel("Searchword")
plt.title("Rise and Fall Percentage by Model and Searchword")
plt.legend(title="Model", loc = 'lower left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plotting RMSE
plt.figure(figsize=(10, 6))
pivot_df_rmse.plot(kind="bar", color=['#1f77b4', '#ff7f0e'], edgecolor="black")  # Different colors for distinction
plt.ylabel("RMSE")
plt.xlabel("Searchword")
plt.title("RMSE by Model and Searchword")
plt.legend(title="Model", loc = 'lower left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()