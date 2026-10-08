import os
import torch


# ============================================================
# TASK 3 - GRADIENT CLIPPING
# ============================================================

# Input directory containing Task 1 gradients
INPUT_DIR = "task1_fl_setup/updates"

# Output directory for protected gradients
OUTPUT_DIR = "task3_defenses/protected_updates"

# Client gradient files
CLIENTS = [
    "client1_gradient.pt",
    "client2_gradient.pt",
    "client3_gradient.pt",
    "client4_gradient.pt"
]

# Clipping thresholds required for Task 3
CLIP_VALUES = [1.0, 0.5, 0.1]


# ============================================================
# Calculate Global L2 Norm
# ============================================================

def calculate_global_norm(gradients):
    """
    Calculate the global L2 norm across all gradient tensors.
    """

    total_squared_norm = 0.0

    for grad in gradients:
        total_squared_norm += torch.sum(
            grad.float() ** 2
        ).item()

    return total_squared_norm ** 0.5


# ============================================================
# Gradient Clipping
# ============================================================

def clip_gradients(gradients, clip_value):
    """
    Clip all gradients using a single global L2 norm.

    Formula:

        scale = min(1, C / ||g||)

        g_clipped = g * scale
    """

    original_norm = calculate_global_norm(gradients)

    scale = min(
        1.0,
        clip_value / (original_norm + 1e-12)
    )

    clipped_gradients = []

    for grad in gradients:
        clipped_gradients.append(
            grad * scale
        )

    clipped_norm = calculate_global_norm(
        clipped_gradients
    )

    return clipped_gradients, original_norm, clipped_norm, scale


# ============================================================
# Main Program
# ============================================================

def main():

    # Create output directory
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print("=" * 70)
    print("TASK 3 - GRADIENT CLIPPING")
    print("=" * 70)

    for client_file in CLIENTS:

        input_path = os.path.join(
            INPUT_DIR,
            client_file
        )

        print("\n" + "-" * 70)
        print(f"Client: {client_file}")
        print("-" * 70)

        # Load original gradient
        gradients = torch.load(
            input_path,
            map_location="cpu",
            weights_only=False
        )

        print(
            f"Number of gradient tensors: "
            f"{len(gradients)}"
        )

        # Test every clipping threshold
        for clip_value in CLIP_VALUES:

            (
                clipped_gradients,
                original_norm,
                clipped_norm,
                scale
            ) = clip_gradients(
                gradients,
                clip_value
            )

            # Generate output filename
            output_filename = client_file.replace(
                ".pt",
                f"_clip_{clip_value}.pt"
            )

            output_path = os.path.join(
                OUTPUT_DIR,
                output_filename
            )

            # Save protected gradient
            torch.save(
                clipped_gradients,
                output_path
            )

            print(
                f"C = {clip_value:<4} | "
                f"Original norm = {original_norm:.6f} | "
                f"Clipped norm = {clipped_norm:.6f} | "
                f"Scale = {scale:.6f}"
            )

    print("\n" + "=" * 70)
    print("Gradient clipping completed successfully.")
    print(f"Protected files saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()