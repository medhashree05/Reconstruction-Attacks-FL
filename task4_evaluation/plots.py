from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

df = pd.read_csv(RESULTS / "comparison_results.csv")

# Average results across the four clients for each configuration.
summary = (
    df.groupby(["Defense", "Parameter"], as_index=False)
    .agg(
        MSE=("MSE", "mean"),
        PSNR=("PSNR", "mean"),
        SSIM=("SSIM", "mean"),
        Attack_Success=("Attack_Success", "mean"),
    )
)

# 1. DP noise level vs average reconstruction error.
dp = summary[summary["Defense"] == "DP Noise"].sort_values("Parameter")

plt.figure(figsize=(8, 5))
plt.plot(dp["Parameter"], dp["MSE"], marker="o")
plt.xlabel("DP Noise Level (sigma)")
plt.ylabel("Average MSE")
plt.title("DP Noise Level vs Reconstruction Error")
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS / "dp_noise_vs_mse.png", dpi=300)
plt.show()

# 2. DP noise level vs average PSNR.
plt.figure(figsize=(8, 5))
plt.plot(dp["Parameter"], dp["PSNR"], marker="o")
plt.xlabel("DP Noise Level (sigma)")
plt.ylabel("Average PSNR (dB)")
plt.title("DP Noise Level vs Reconstruction Quality")
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS / "dp_noise_vs_psnr.png", dpi=300)
plt.show()

# 3. Compare all defenses using average PSNR.
defense_order = [
    ("No Defense", 0.0, "No Defense"),
    ("Clipping", 1.0, "Clip 1.0"),
    ("Clipping", 0.5, "Clip 0.5"),
    ("Clipping", 0.1, "Clip 0.1"),
    ("DP Noise", 0.01, "DP 0.01"),
    ("DP Noise", 0.05, "DP 0.05"),
    ("DP Noise", 0.1, "DP 0.1"),
    ("DP Noise", 0.5, "DP 0.5"),
]

labels = []
values = []

for defense, parameter, label in defense_order:
    row = summary[
        (summary["Defense"] == defense)
        & (summary["Parameter"] == parameter)
    ]

    if not row.empty:
        labels.append(label)
        values.append(row["PSNR"].iloc[0])

plt.figure(figsize=(10, 5))
plt.bar(labels, values)
plt.xlabel("Experiment")
plt.ylabel("Average PSNR (dB)")
plt.title("Defense Method vs Reconstruction Quality")
plt.xticks(rotation=30, ha="right")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig(RESULTS / "defense_vs_psnr.png", dpi=300)
plt.show()

# 4. Attack success rate for each defense.
success_values = []

for defense, parameter, label in defense_order:
    row = summary[
        (summary["Defense"] == defense)
        & (summary["Parameter"] == parameter)
    ]

    if not row.empty:
        success_values.append(
            row["Attack_Success"].iloc[0] * 100
        )

plt.figure(figsize=(10, 5))
plt.bar(labels, success_values)
plt.xlabel("Experiment")
plt.ylabel("Attack Success Rate (%)")
plt.title("Defense Method vs DLG Attack Success Rate")
plt.xticks(rotation=30, ha="right")
plt.ylim(0, 100)
plt.grid(axis="y")
plt.tight_layout()
plt.savefig(RESULTS / "defense_vs_attack_success.png", dpi=300)
plt.show()

# Save averages for the report.
summary.to_csv(RESULTS / "average_comparison_results.csv", index=False)

print("\nAverage results across all clients:")
print(summary.to_string(index=False))
print("\nGraphs and average results saved in:", RESULTS)

