import os
import torch


# ============================================================
# TASK 3 - DIFFERENTIAL PRIVACY
# ============================================================

INPUT_DIR = "task1_fl_setup/updates"
OUTPUT_DIR = "task3_defenses/protected_updates"

CLIENTS = [
    "client1_gradient.pt",
    "client2_gradient.pt",
    "client3_gradient.pt",
    "client4_gradient.pt"
]

# DP requires clipping before adding noise
CLIP_VALUE = 1.0

# Noise levels required by Task 3
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
# Gradient Clipping
# ============================================================

def clip_gradients(gradients, clip_value):

    original_norm = calculate_global_norm(
        gradients
    )

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

    return (
        clipped_gradients,
        original_norm,
        clipped_norm,
        scale
    )


# ============================================================
# Add Gaussian Differential Privacy Noise
# ============================================================

def add_gaussian_noise(
    gradients,
    noise_level
):

    noisy_gradients = []

    for grad in gradients:

        noise = torch.randn_like(
            grad
        ) * noise_level

        noisy_gradients.append(
            grad + noise
        )

    return noisy_gradients


# ============================================================
# Main
# ============================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print("=" * 70)
    print("TASK 3 - DIFFERENTIAL PRIVACY")
    print("=" * 70)

    for client_file in CLIENTS:

        input_path = os.path.join(
            INPUT_DIR,
            client_file
        )

        gradients = torch.load(
            input_path,
            map_location="cpu",
            weights_only=False
        )

        print("\n" + "-" * 70)
        print(f"Client: {client_file}")
        print("-" * 70)

        # ----------------------------------------------------
        # Step 1: Clip gradient
        # ----------------------------------------------------

        (
            clipped_gradients,
            original_norm,
            clipped_norm,
            scale
        ) = clip_gradients(
            gradients,
            CLIP_VALUE
        )

        print(
            f"Original norm : {original_norm:.6f}"
        )

        print(
            f"Clipped norm  : {clipped_norm:.6f}"
        )

        print(
            f"Clip value    : {CLIP_VALUE}"
        )

        # ----------------------------------------------------
        # Step 2: Add different noise levels
        # ----------------------------------------------------

        for noise_level in NOISE_LEVELS:

            protected_gradients = add_gaussian_noise(
                clipped_gradients,
                noise_level
            )

            output_filename = client_file.replace(
                ".pt",
                f"_dp_{noise_level}.pt"
            )

            output_path = os.path.join(
                OUTPUT_DIR,
                output_filename
            )

            torch.save(
                protected_gradients,
                output_path
            )

            protected_norm = calculate_global_norm(
                protected_gradients
            )

            print(
                f"Noise = {noise_level:<5} | "
                f"Protected norm = "
                f"{protected_norm:.6f}"
            )

    print("\n" + "=" * 70)
    print("Differential Privacy completed successfully.")
    print(
        f"Protected files saved to: {OUTPUT_DIR}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()