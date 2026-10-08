import os
import csv
import json
import torch

from task1_fl_setup.model import LeNet
from task2_reconstruction_attack.dlg_attack import DLGAttacker
from task2_reconstruction_attack.metrics import evaluate_reconstruction


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "task1_fl_setup/models/global_model_init.pth"
UPDATES_DIR = "task1_fl_setup/updates"
PROTECTED_DIR = "task3_defenses/protected_updates"
OUTPUT_DIR = "task3_defenses/results"

DEVICE = "cpu"

MAX_ITER = 100
LR = 1.0
OPTIMIZER = "lbfgs"
COST_FN = "l2"
LABEL_STRATEGY = "joint"
TOLERANCE = 1e-6
SEED = 42

IMG_SHAPE = (1, 1, 28, 28)


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

def load_ground_truth(client_id):

    path = os.path.join(
        UPDATES_DIR,
        f"client{client_id}_ground_truth.pt"
    )

    data = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    if not isinstance(data, dict):
        raise ValueError(
            f"Unexpected ground-truth format: {type(data)}"
        )

    true_image = data["image"]
    true_label = data["label"]

    return true_image, true_label


# ============================================================
# FIND PROTECTED FILES
# ============================================================

def get_protected_files():

    files = []

    for filename in os.listdir(PROTECTED_DIR):

        if not filename.endswith(".pt"):
            continue

        if "_clip_" in filename or "_dp_" in filename:
            files.append(filename)

    return sorted(files)


# ============================================================
# GET DEFENSE INFORMATION
# ============================================================

def get_defense_info(filename):

    if "_clip_" in filename:

        value = filename.split("_clip_")[1]
        value = value.replace(".pt", "")

        return "clipping", float(value)

    if "_dp_" in filename:

        value = filename.split("_dp_")[1]
        value = value.replace(".pt", "")

        return "dp", float(value)

    return "unknown", None


# ============================================================
# GET CLIENT ID
# ============================================================

def get_client_id(filename):

    if filename.startswith("client1_"):
        return 1

    if filename.startswith("client2_"):
        return 2

    if filename.startswith("client3_"):
        return 3

    if filename.startswith("client4_"):
        return 4

    return None


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 75)
    print("TASK 3 - DEFENSE RECONSTRUCTION EVALUATION")
    print("=" * 75)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading model...")

    model = LeNet()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # Find protected files
    # --------------------------------------------------------

    protected_files = get_protected_files()

    print(
        f"\nProtected gradient files found: "
        f"{len(protected_files)}"
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results = []

    # --------------------------------------------------------
    # Run DLG on every protected gradient
    # --------------------------------------------------------

    for index, filename in enumerate(
        protected_files,
        start=1
    ):

        client_id = get_client_id(filename)

        defense, parameter = get_defense_info(
            filename
        )

        if client_id is None:
            print(
                f"\nSkipping unknown file: {filename}"
            )
            continue

        print("\n" + "=" * 75)
        print(
            f"[{index}/{len(protected_files)}] "
            f"{filename}"
        )
        print(f"Client    : {client_id}")
        print(f"Defense   : {defense}")
        print(f"Parameter : {parameter}")
        print("=" * 75)

        # ----------------------------------------------------
        # Load protected gradient
        # ----------------------------------------------------

        gradient_path = os.path.join(
            PROTECTED_DIR,
            filename
        )

        target_gradients = torch.load(
            gradient_path,
            map_location="cpu",
            weights_only=False
        )

        # ----------------------------------------------------
        # Load ground truth ONLY for evaluation
        # ----------------------------------------------------

        true_image, true_label = load_ground_truth(
            client_id
        )

        # ----------------------------------------------------
        # Create attacker
        # ----------------------------------------------------

        attacker = DLGAttacker(
            model=model,
            device=DEVICE,
            optimizer_type=OPTIMIZER,
            lr=LR,
            max_iter=MAX_ITER,
            cost_fn=COST_FN,
            label_strategy=LABEL_STRATEGY,
            tolerance=TOLERANCE,
            seed=SEED
        )

        # ----------------------------------------------------
        # Run reconstruction attack
        # ----------------------------------------------------

        print("\nRunning reconstruction attack...")

        reconstruction = attacker.attack(
            target_gradients=target_gradients,
            img_shape=IMG_SHAPE,
            true_image=true_image,
            true_label=true_label,
            log_interval=20,
            progression_steps=[
                0,
                5,
                10,
                20,
                50,
                MAX_ITER
            ]
        )

        # ----------------------------------------------------
        # Evaluate reconstruction
        # ----------------------------------------------------

        metrics = evaluate_reconstruction(
            reconstruction["reconstructed_image"],
            true_image,
            reconstruction["reconstructed_label"],
            true_label
        )

        # ----------------------------------------------------
        # Final gradient loss
        # ----------------------------------------------------

        if reconstruction["loss_history"]:

            final_grad_loss = (
                reconstruction["loss_history"][-1]
            )

        else:

            final_grad_loss = None

        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        result = {

            "client_id": client_id,

            "defense": defense,

            "parameter": parameter,

            "filename": filename,

            "true_label": int(
                true_label.item()
            ),

            "pred_label": int(
                reconstruction[
                    "reconstructed_label"
                ]
            ),

            "label_match": metrics[
                "label_match"
            ],

            "mse": metrics["mse"],

            "psnr": metrics["psnr"],

            "ssim": metrics["ssim"],

            "success": metrics["success"],

            "iterations": reconstruction[
                "iterations"
            ],

            "final_grad_loss": final_grad_loss,

            "duration_sec": reconstruction[
                "duration"
            ]
        }

        results.append(result)

        # ----------------------------------------------------
        # Print result
        # ----------------------------------------------------

        print("\nRESULT")
        print("-" * 50)

        print(
            f"MSE           : "
            f"{result['mse']}"
        )

        print(
            f"PSNR          : "
            f"{result['psnr']}"
        )

        print(
            f"SSIM          : "
            f"{result['ssim']}"
        )

        print(
            f"True label    : "
            f"{result['true_label']}"
        )

        print(
            f"Pred label    : "
            f"{result['pred_label']}"
        )

        print(
            f"Label match   : "
            f"{result['label_match']}"
        )

        print(
            f"Attack success: "
            f"{result['success']}"
        )

        print(
            f"Iterations    : "
            f"{result['iterations']}"
        )

        print(
            f"Duration      : "
            f"{result['duration_sec']:.2f}s"
        )

    # ========================================================
    # SAVE CSV
    # ========================================================

    csv_path = os.path.join(
        OUTPUT_DIR,
        "reconstruction_defense_results.csv"
    )

    fieldnames = [

        "client_id",

        "defense",

        "parameter",

        "filename",

        "true_label",

        "pred_label",

        "label_match",

        "mse",

        "psnr",

        "ssim",

        "success",

        "iterations",

        "final_grad_loss",

        "duration_sec"
    ]

    with open(
        csv_path,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(results)

    # ========================================================
    # SAVE JSON
    # ========================================================

    json_path = os.path.join(
        OUTPUT_DIR,
        "reconstruction_defense_results.json"
    )

    with open(
        json_path,
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 75)
    print(
        "TASK 3 RECONSTRUCTION "
        "EVALUATION COMPLETED"
    )
    print("=" * 75)

    print(
        f"\nTotal experiments: "
        f"{len(results)}"
    )

    print(
        f"\nCSV : {csv_path}"
    )

    print(
        f"JSON: {json_path}"
    )

    print("=" * 75)


if __name__ == "__main__":
    main()