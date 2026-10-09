
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

experiments = [
    ("no_defense", "No Defense", 0),
    ("clip_1.0", "Clipping", 1.0),
    ("clip_0.5", "Clipping", 0.5),
    ("clip_0.1", "Clipping", 0.1),
    ("dp_0.0", "DP Noise", 0.0),
    ("dp_0.01", "DP Noise", 0.01),
    ("dp_0.05", "DP Noise", 0.05),
    ("dp_0.1", "DP Noise", 0.1),
    ("dp_0.5", "DP Noise", 0.5),
]

records = []

for folder, defense, parameter in experiments:
    csv_path = RESULTS / folder / "reconstruction_summary.csv"

    if not csv_path.exists():
        print(f"Missing: {csv_path}")
        continue

    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        records.append({
            "Defense": defense,
            "Parameter": parameter,
            "Client": int(row["client_id"]),
            "MSE": row["mse"],
            "PSNR": row["psnr"],
            "SSIM": row["ssim"],
            "Attack_Success": row["success"],
            "Iterations": row["iterations"],
        })

comparison = pd.DataFrame(records)

output = RESULTS / "comparison_results.csv"
comparison.to_csv(output, index=False)

print("\nPART 4 COMPARISON RESULTS")
print("=" * 90)
print(comparison.to_string(index=False))
print(f"\nSaved to: {output}")
