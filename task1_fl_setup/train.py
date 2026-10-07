"""Task 1: FL setup. Run from inside task1_fl_setup/:  python train.py"""
import json
import os
import torch

from dataset import load_mnist, split_clients, test_loader
from model import LeNet
from client import Client
from server import fedavg, evaluate

NUM_CLIENTS = 4
ROUNDS = 20
LOCAL_EPOCHS = 2
LR = 0.01
SEED = 0
TARGET_INDEX = 0          # which of each client's images is the "private" target
device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED)
os.makedirs("models", exist_ok=True)
os.makedirs("updates", exist_ok=True)

train_set, test_set = load_mnist()
tloader = test_loader(test_set)
clients = [Client(i + 1, ds, device) for i, ds in enumerate(split_clients(train_set, NUM_CLIENTS))]

global_model = LeNet().to(device)
torch.save(global_model.state_dict(), "models/global_model_init.pth")
results = {"round_accuracy": [], "initial_accuracy": evaluate(global_model, tloader, device)}

# ---- Save the attack targets from the INITIAL global model ----
# Attacker (honest-but-curious server) knows: model weights, architecture, loss,
# and the gradient. The ground truth image/label is saved separately and is used
# ONLY to score the attack in Task 2/4, never given to the attacker.
for c in clients:
    grads, img, lbl = c.single_sample_gradient(global_model.cpu(), TARGET_INDEX)
    torch.save(grads, f"updates/client{c.cid}_gradient.pt")
    torch.save({"image": img, "label": lbl}, f"updates/client{c.cid}_ground_truth.pt")
global_model.to(device)

# ---- Federated training ----
for r in range(1, ROUNDS + 1):
    updates = [c.local_train(global_model, LOCAL_EPOCHS, LR) for c in clients]
    fedavg(global_model, [{k: v for k, v in u.items()} for u in updates])
    acc = evaluate(global_model, tloader, device)
    results["round_accuracy"].append(acc)
    print(f"Round {r:2d}/{ROUNDS}  test acc = {acc:.4f}")

# Save last round's model updates (weight deltas) per client
for c, u in zip(clients, updates):
    torch.save(u, f"updates/client{c.cid}_update.pt")

torch.save(global_model.state_dict(), "models/global_model.pth")
results["baseline_accuracy"] = results["round_accuracy"][-1]
results["config"] = dict(num_clients=NUM_CLIENTS, rounds=ROUNDS, local_epochs=LOCAL_EPOCHS, lr=LR, seed=SEED)
with open("models/baseline_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Baseline accuracy:", results["baseline_accuracy"])
