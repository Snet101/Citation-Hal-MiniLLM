import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read data
df = pd.read_csv("hallucination_summary.csv")

# Aesthetic setup
sns.set(style="whitegrid", font_scale=1.2)

plt.figure(figsize=(6, 4))
sns.lineplot(
    data=df,
    x="variant",
    y="hallucination_rate_%",
    hue="source_path",
    marker="o",
    linewidth=2.5,
)

plt.title("Hallucination Rate by Prompt Type", fontsize=14, weight="bold")
plt.xlabel("Prompt Variant", fontsize=12)
plt.ylabel("Hallucination Rate (%)", fontsize=12)
plt.ylim(0, 100)
plt.legend(title="Prompt Category", loc="lower left", frameon=True)
plt.tight_layout()
plt.savefig("hallucination_rate_plot.png", dpi=300)
plt.show()

