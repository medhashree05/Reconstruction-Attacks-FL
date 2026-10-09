
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "task4_evaluation" / "results"

results = pd.read_csv(RESULTS_DIR / "model_accuracy_results.csv")
history = pd.read_csv(RESULTS_DIR / "model_accuracy_history.csv")

# Plot 1: Final model accuracy by defense
plt.figure(figsize=(10, 6))
plt.bar(results["experiment"], results["final_accuracy_percent"])
plt.ylabel("Final Test Accuracy (%)")
plt.xlabel("Defense Configuration")
plt.title("Model Accuracy Under Different Defense Mechanisms")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "model_accuracy_comparison.png", dpi=300)
plt.show()

# Plot 2: Accuracy across training rounds
plt.figure(figsize=(10, 6))
for experiment, group in history.groupby("experiment"):
    plt.plot(group["round"], group["accuracy"] * 100, marker=".", label=experiment)

plt.xlabel("Federated Learning Round")
plt.ylabel("Test Accuracy (%)")
plt.title("Federated Learning Accuracy Across Rounds")
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "accuracy_over_rounds.png", dpi=300)
plt.show()

print("Saved:")
print(RESULTS_DIR / "model_accuracy_comparison.png")
print(RESULTS_DIR / "accuracy_over_rounds.png")
