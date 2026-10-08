import os
import csv
import torch


# ============================================================
# TASK 3 - VALIDATION
# ============================================================

INPUT_DIR = "task1_fl_setup/updates"
OUTPUT_DIR = "task3_defenses/protected_updates"
RESULTS_DIR = "task3_defenses"

CLIENTS = [
    "client1",
    "client2",
    "client3",
    "client4"
]

CLIP_VALUES = [1.0, 0.5, 0.1]

NOISE_LEVELS = [
    0.0,
    0.01,
    0.05,
    0.1,
    0.5
]


# ============================================================
# Calculate Global L2 Norm
# ============================================================

def calculate_global_norm(gradients):

    total_squared_norm = 0.0

    for grad in gradients:
        total_squared_norm += torch.sum(
            grad.float() ** 2
        ).item()

    return total_squared_norm ** 0.5


# ============================================================
# Load Gradient
# ============================================================

def load_gradient(path):

    return torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )


# ============================================================
# Main
# ============================================================

def main():

    clipping_results = []
    dp_results = []

    print("=" * 80)
    print("TASK 3 - VALIDATION RESULTS")
    print("=" * 80)

    # ========================================================
    # Original gradient norms
    # ========================================================

    original_norms = {}

    for client in CLIENTS:

        path = os.path.join(
            INPUT_DIR,
            f"{client}_gradient.pt"
        )

        gradients = load_gradient(path)

        norm = calculate_global_norm(gradients)

        original_norms[client] = norm

    # ========================================================
    # VALIDATE GRADIENT CLIPPING
    # ========================================================

    print("\n")
    print("=" * 80)
    print("GRADIENT CLIPPING")
    print("=" * 80)

    for client in CLIENTS:

        original_norm = original_norms[client]

        for clip_value in CLIP_VALUES:

            filename = (
                f"{client}_gradient_clip_{clip_value}.pt"
            )

            path = os.path.join(
                OUTPUT_DIR,
                filename
            )

            gradients = load_gradient(path)

            protected_norm = calculate_global_norm(
                gradients
            )

            scale = min(
                1.0,
                clip_value / (original_norm + 1e-12)
            )

            result = {
                "Client": client,
                "Clip_Value": clip_value,
                "Original_Norm": original_norm,
                "Protected_Norm": protected_norm,
                "Scale": scale
            }

            clipping_results.append(result)

            print(
                f"{client:<8} | "
                f"C={clip_value:<4} | "
                f"Original={original_norm:.6f} | "
                f"Protected={protected_norm:.6f} | "
                f"Scale={scale:.6f}"
            )

    # ========================================================
    # VALIDATE DIFFERENTIAL PRIVACY
    # ========================================================

    print("\n")
    print("=" * 80)
    print("DIFFERENTIAL PRIVACY")
    print("=" * 80)

    for client in CLIENTS:

        original_norm = original_norms[client]

        # DP always clips to C=1.0
        clipped_norm = 1.0

        for noise_level in NOISE_LEVELS:

            filename = (
                f"{client}_gradient_dp_{noise_level}.pt"
            )

            path = os.path.join(
                OUTPUT_DIR,
                filename
            )

            gradients = load_gradient(path)

            protected_norm = calculate_global_norm(
                gradients
            )

            result = {
                "Client": client,
                "Clip_Value": 1.0,
                "Noise_Level": noise_level,
                "Original_Norm": original_norm,
                "Clipped_Norm": clipped_norm,
                "Protected_Norm": protected_norm
            }

            dp_results.append(result)

            print(
                f"{client:<8} | "
                f"Noise={noise_level:<5} | "
                f"Original={original_norm:.6f} | "
                f"Clipped=1.000000 | "
                f"Protected={protected_norm:.6f}"
            )

    # ========================================================
    # SAVE CLIPPING CSV
    # ========================================================

    clipping_csv = os.path.join(
        RESULTS_DIR,
        "clipping_results.csv"
    )

    with open(
        clipping_csv,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "Client",
                "Clip_Value",
                "Original_Norm",
                "Protected_Norm",
                "Scale"
            ]
        )

        writer.writeheader()
        writer.writerows(clipping_results)

    # ========================================================
    # SAVE DP CSV
    # ========================================================

    dp_csv = os.path.join(
        RESULTS_DIR,
        "dp_results.csv"
    )

    with open(
        dp_csv,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "Client",
                "Clip_Value",
                "Noise_Level",
                "Original_Norm",
                "Clipped_Norm",
                "Protected_Norm"
            ]
        )

        writer.writeheader()
        writer.writerows(dp_results)

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n")
    print("=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)

    print(
        f"\nClipping results saved to:"
        f"\n{clipping_csv}"
    )

    print(
        f"\nDP results saved to:"
        f"\n{dp_csv}"
    )

    print(
        "\nTotal clipping experiments:",
        len(clipping_results)
    )

    print(
        "Total DP experiments:",
        len(dp_results)
    )

    print(
        "\nTotal experiments:",
        len(clipping_results) + len(dp_results)
    )


if __name__ == "__main__":
    main()