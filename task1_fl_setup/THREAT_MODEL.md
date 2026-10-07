# Threat Model
**Attacker:** honest-but-curious central server. It follows the FedAvg protocol correctly
but tries to reconstruct clients' private training data from what it receives.

**Attacker knows:** model architecture, current global weights, loss function (cross-entropy),
client gradients / model updates. Labels may be known or inferred (DLG can recover them).

**Attacker does NOT have:** the raw private images (ground-truth files are used only for scoring).

**Goal:** recover the private MNIST image (and label) behind a client's gradient.
