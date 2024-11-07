import matplotlib.pyplot as plt
import pandas as pd

# Updated data based on the latest details
updated_data = {
    "Searchword": ["bitcoin", "bitcoin", "ethereum", "ethereum", "tether", "tether", 
                   "cardano", "cardano", "polkadot", "polkadot", "vechain", "vechain"],
    "Model": ["ARMA", "ARMAX", "ARMA", "ARMAX", "ARMA", "ARMAX", 
              "ARMA", "ARMAX", "ARMA", "ARMAX", "ARMA", "ARMAX"],
    "Rise and fall (%)": [78.12, 77.08, 75.00, 78.12, 75.08, 75.00, 
                          73.96, 69.79, 73.96, 75.00, 71.88, 73.96],
    "RMSE": [0.292, 0.278, 0.278, 0.262, 0.230, 0.236, 
             0.310, 0.338, 0.310, 0.289, 0.275, 0.275]
}

# Create DataFrame
df_updated = pd.DataFrame(updated_data)

# Pivot the DataFrame to format it for grouped bar plotting
pivot_df_rise_and_fall = df_updated.pivot(index="Searchword", columns="Model", values="Rise and fall (%)")
pivot_df_rmse = df_updated.pivot(index="Searchword", columns="Model", values="RMSE")

# Plotting Rise and Fall Percentage
plt.figure(figsize=(10, 6))
pivot_df_rise_and_fall.plot(kind="bar", edgecolor="black", color=['#1f77b4', '#ff7f0e'])
plt.ylabel("Rise and Fall (%)")
plt.xlabel("Searchword")
plt.title("Rise and Fall percentage by model and searchword")
plt.legend(title="Model", loc = "lower left")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plotting RMSE
plt.figure(figsize=(10, 6))
pivot_df_rmse.plot(kind="bar", edgecolor="black", color=['#1f77b4', '#ff7f0e'])
plt.ylabel("RMSE")
plt.xlabel("Searchword")
plt.title("RMSE by model and searchword")
plt.legend(title="Model", loc = "lower left")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
