from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent

ORIGINAL_UPDATES = ROOT / "task1_fl_setup" / "updates"
PROTECTED_UPDATES = ROOT / "task3_defenses" / "protected_updates"
RESULTS = ROOT / "task4_evaluation" / "results"

ATTACK_SCRIPT = ROOT / "task2_reconstruction_attack" / "reconstruction.py"
MODEL_PATH = ROOT / "task1_fl_setup" / "models" / "global_model_init.pth"

CLIENTS = [1, 2, 3, 4]


def run_experiment(name, gradient_pattern):
    experiment_dir = RESULTS / name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    for client_id in CLIENTS:
        if gradient_pattern == "original":
            gradient_file = (
                ORIGINAL_UPDATES / f"client{client_id}_gradient.pt"
            )
        else:
            gradient_file = (
                PROTECTED_UPDATES
                / gradient_pattern.format(client=client_id)
            )

        if not gradient_file.exists():
            raise FileNotFoundError(
                f"Missing gradient file: {gradient_file}"
            )

        shutil.copy2(
            gradient_file,
            experiment_dir / f"client{client_id}_gradient.pt"
        )

        shutil.copy2(
            ORIGINAL_UPDATES / f"client{client_id}_ground_truth.pt",
            experiment_dir / f"client{client_id}_ground_truth.pt"
        )

    command = [
        sys.executable,
        str(ATTACK_SCRIPT),
        "--clients", *map(str, CLIENTS),
        "--updates-dir", str(experiment_dir),
        "--output-dir", str(experiment_dir),
        "--model-path", str(MODEL_PATH),
    ]

    print(f"\n{'=' * 60}\nExperiment: {name}\n{'=' * 60}")
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    run_experiment("no_defense", "original")

    for threshold in [1.0, 0.5, 0.1]:
        run_experiment(
            f"clip_{threshold}",
            "client{client}_gradient_clip_" + str(threshold) + ".pt"
        )

    for sigma in [0.0, 0.01, 0.05, 0.1, 0.5]:
        run_experiment(
            f"dp_{sigma}",
            "client{client}_gradient_dp_" + str(sigma) + ".pt"
        )

    print("\nAll four-client experiments completed.")


if __name__ == "__main__":
    main()
