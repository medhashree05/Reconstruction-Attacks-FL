
import copy
import csv
from pathlib import Path

import torch

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "task1_fl_setup"))

from dataset import load_mnist, split_clients, test_loader
from model import LeNet
from client import Client
from server import fedavg, evaluate

OUTPUT_DIR = ROOT / "task4_evaluation" / "results"

NUM_CLIENTS = 4
ROUNDS = 20
LOCAL_EPOCHS = 2
LR = 0.01
SEED = 0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def clip_update(update, clip_value):
    """Clip the global L2 norm of one client's complete model update."""
    norm_squared = sum(
        tensor.float().pow(2).sum().item()
        for tensor in update.values()
    )
    norm = norm_squared ** 0.5

    scale = min(1.0, clip_value / (norm + 1e-12))

    clipped = {
        key: tensor * scale
        for key, tensor in update.items()
    }

    return clipped


def protect_update(update, defense, parameter):
    """Apply protection to a client update before server aggregation."""
    if defense == "none":
        return update

    if defense == "clip":
        return clip_update(update, parameter)

    if defense == "dp":
        # Clip first, then add Gaussian noise.
        clipped = clip_update(update, 1.0)

        return {
            key: tensor + torch.randn_like(tensor) * parameter
            for key, tensor in clipped.items()
        }

    raise ValueError(f"Unknown defense: {defense}")


def run_experiment(name, defense, parameter):
    print("\n" + "=" * 65)
    print(f"Experiment: {name}")
    print("=" * 65)

    # Reset initialization for a consistent starting point.
    torch.manual_seed(SEED)

    train_set, test_set = load_mnist()
    test_data = test_loader(test_set)

    client_datasets = split_clients(train_set, NUM_CLIENTS)
    clients = [
        Client(i + 1, dataset, DEVICE)
        for i, dataset in enumerate(client_datasets)
    ]

    model = LeNet().to(DEVICE)
    initial_accuracy = evaluate(model, test_data, DEVICE)

    history = []

    print(f"Device: {DEVICE}")
    print(f"Initial test accuracy: {initial_accuracy:.4f}")

    for round_num in range(1, ROUNDS + 1):
        # Train each client locally using the existing Task 1 code.
        updates = [
            client.local_train(model, LOCAL_EPOCHS, LR)
            for client in clients
        ]

        # Protect each client update before FedAvg.
        protected_updates = [
            protect_update(update, defense, parameter)
            for update in updates
        ]

        fedavg(model, protected_updates)

        accuracy = evaluate(model, test_data, DEVICE)
        history.append(accuracy)

        print(
            f"Round {round_num:02d}/{ROUNDS} | "
            f"Test accuracy: {accuracy:.4f}"
        )

    final_accuracy = history[-1]

    print(f"\nFinal test accuracy: {final_accuracy:.4f}")
    print(f"Accuracy percentage: {final_accuracy * 100:.2f}%")

    return {
        "experiment": name,
        "defense": defense,
        "parameter": parameter,
        "initial_accuracy": initial_accuracy,
        "final_accuracy": final_accuracy,
        "final_accuracy_percent": final_accuracy * 100,
        "accuracy_history": history,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    experiments = [
        ("No Defense", "none", 0.0),
        ("Clip 1.0", "clip", 1.0),
        ("Clip 0.5", "clip", 0.5),
        ("Clip 0.1", "clip", 0.1),
        ("DP 0.01", "dp", 0.01),
        ("DP 0.05", "dp", 0.05),
        ("DP 0.1", "dp", 0.1),
        ("DP 0.5", "dp", 0.5),
    ]

    results = []

    for name, defense, parameter in experiments:
        results.append(
            run_experiment(name, defense, parameter)
        )

    csv_path = OUTPUT_DIR / "model_accuracy_results.csv"

    with open(csv_path, "w", newline="") as file:
        fields = [
            "experiment",
            "defense",
            "parameter",
            "initial_accuracy",
            "final_accuracy",
            "final_accuracy_percent",
        ]

        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for result in results:
            writer.writerow({
                key: result[key]
                for key in fields
            })

    # Save round-by-round results for plotting.
    history_path = OUTPUT_DIR / "model_accuracy_history.csv"

    with open(history_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["experiment", "round", "accuracy"])

        for result in results:
            for round_num, accuracy in enumerate(
                result["accuracy_history"], start=1
            ):
                writer.writerow([
                    result["experiment"],
                    round_num,
                    accuracy,
                ])

    print("\n" + "=" * 65)
    print("FINAL MODEL ACCURACY COMPARISON")
    print("=" * 65)

    for result in results:
        print(
            f"{result['experiment']:<15} "
            f"{result['final_accuracy_percent']:.2f}%"
        )

    print(f"\nResults saved to: {csv_path}")
    print(f"Training history saved to: {history_path}")


if __name__ == "__main__":
    main()
