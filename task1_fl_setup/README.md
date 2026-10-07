# Task 1: Federated Learning Setup and Threat Model

Part of **Reconstruction Attacks in Federated Learning: Privacy Risks and Defense Mechanisms**.

This task builds a plain FL system on MNIST (no attack, no defense) and saves the
gradients and model updates that Task 2 (attack) and Task 3 (defenses) work on.

**Baseline test accuracy: 96.48%** (20 rounds, FedAvg, 4 clients)

## Quick start

```bash
pip install -r ../requirements.txt
cd task1_fl_setup
python train.py
```

Run it from inside `task1_fl_setup/`: paths like `models/` and `updates/` are relative.
MNIST downloads to `./data/` on the first run (git-ignored). Re-running overwrites
everything in `models/` and `updates/`.

## Files

| File | Purpose |
|---|---|
| `dataset.py` | Loads MNIST and splits it IID across clients (pixels in [0,1], no normalization) |
| `model.py` | `LeNet`: small CNN with **Tanh** activations |
| `client.py` | `Client.local_train()` (returns weight delta) and `Client.single_sample_gradient()` (attack target) |
| `server.py` | `fedavg()` and `evaluate()` |
| `train.py` | Runs the whole pipeline and saves outputs |
| `THREAT_MODEL.md` | Attacker definition |

## Configuration (top of `train.py`)

| Setting | Value |
|---|---|
| Clients | 4 (2000 MNIST images each, disjoint, IID) |
| Rounds | 20 |
| Local epochs | 2 |
| Optimizer | SGD, lr = 0.01, momentum = 0.9 (set in `client.py`) |
| Seed | 0 (do not change: Tasks 2 to 4 rely on identical initial weights and targets) |
| Target image | index 0 of each client's data (`TARGET_INDEX`) |

## Saved outputs

```
models/
  global_model_init.pth    initial (untrained) global model; gradients below were computed on THIS model
  global_model.pth         final global model after 20 rounds
  baseline_results.json    accuracy per round, baseline accuracy, config
updates/
  clientK_gradient.pt      single-image gradient (the Task 2 attack target)
  clientK_ground_truth.pt  the true image and label (for scoring ONLY)
  clientK_update.pt        real last-round FedAvg weight delta (K = 1..4)
```

### File formats

- **`clientK_gradient.pt`**: a `list` of 8 tensors, in the order of `model.parameters()`
  (conv weight/bias x3, then fc weight/bias). Loss is cross-entropy on one image.
- **`clientK_ground_truth.pt`**: `{"image": Tensor[1,1,28,28] in [0,1], "label": Tensor[1]}`.
- **`clientK_update.pt`**: a `dict` with the same keys as `model.state_dict()`;
  each value is `local_weights - global_weights` after local training.

### Loading example (for Task 2)

```python
import torch
from model import LeNet

model = LeNet()
model.load_state_dict(torch.load("models/global_model_init.pth"))   # the INITIAL model

target_grad = torch.load("updates/client1_gradient.pt")             # attacker input
gt = torch.load("updates/client1_ground_truth.pt")                  # scoring only
true_img, true_label = gt["image"], gt["label"]
```

## Notes for the next tasks

- **Task 2 (attack):** use `global_model_init.pth` with `clientK_gradient.pt`. Using the
  final model with these gradients will not match. The attacker may use the model, the
  gradient and the loss. `clientK_ground_truth.pt` is only for computing MSE/PSNR and must
  never be given to the attack.
- **Task 3 (defenses):** clip and add noise to `clientK_gradient.pt` (a list of tensors, so
  compute the global L2 norm across all of them) and, if needed, to `clientK_update.pt`.
  Reuse `Client` and `fedavg()` instead of writing a new FL system.
- **Task 4 (evaluation):** accuracy under defense = rerun `train.py`'s loop with clipped/noisy
  updates and compare against the 96.48% baseline.
- Tanh is used instead of Sigmoid on purpose: it is still smooth and twice-differentiable
  (needed for DLG), but a Sigmoid version stayed at about 10% accuracy (vanishing gradients).
- ReLU would break DLG, so do not swap it in.

## Threat model (summary)

Honest-but-curious server: follows FedAvg, sees the model, the loss and client
gradients/updates, and tries to reconstruct private images. It never receives raw images.
See `THREAT_MODEL.md`.