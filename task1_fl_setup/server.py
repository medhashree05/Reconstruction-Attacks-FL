import torch


def fedavg(global_model, client_updates, weights=None):
    """Federated Averaging: new_global = global + weighted mean of client deltas."""
    n = len(client_updates)
    weights = weights or [1.0 / n] * n
    state = global_model.state_dict()
    for k in state:
        avg = sum(w * u[k].float() for w, u in zip(weights, client_updates))
        state[k] = (state[k].float() + avg).to(state[k].dtype)
    global_model.load_state_dict(state)
    return global_model


@torch.no_grad()
def evaluate(model, loader, device="cpu"):
    model.eval().to(device)
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total
