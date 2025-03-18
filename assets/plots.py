import pandas as pd
import matplotlib.pyplot as plt

# Load CSV file
# df = pd.read_csv("lr/lr.csv")
# df = pd.read_csv("loss/model2_loss.csv")
df = pd.read_csv("consis_weights/weight.csv")

# Create the plot
plt.figure(figsize=(6, 4), dpi=300)
# plt.plot(df["Step"], df["Value"], color="#9e0100", linewidth=3)
# plt.plot(df["Step"], df["Value"], color="#1d80bc", linewidth=2)
plt.plot(df["Step"], df["Value"], color="#288c8c", linewidth=3)

# Labels
plt.xlabel("Step", fontsize=12)
# plt.ylabel("LR", fontsize=12)
# plt.ylabel("Model 2 Loss", fontsize=12)
plt.ylabel("Consistency Weight", fontsize=12)

# Grid and layout adjustments
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

# Save the plot
# plt.savefig("lr_plot.png", dpi=300)
# plt.savefig("loss.png", dpi=300)
plt.savefig("weight.png", dpi=300)
plt.show()
