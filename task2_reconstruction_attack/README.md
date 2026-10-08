# Task 2: Reconstruction Attack (Deep Leakage from Gradients)

Part of **Reconstruction Attacks in Federated Learning: Privacy Risks and Defense Mechanisms**.

## Overview

Task 2 implements the **Deep Leakage from Gradients (DLG)** reconstruction attack to demonstrate gradient leakage in Federated Learning. Given only the shared gradients from clients and the public model architecture / initial weights, the honest-but-curious server reconstructs the private client training images and labels with high pixel-level fidelity.

---

## Threat Model (Task 2)

- **Attacker:** Honest-but-curious central server. Follows the FedAvg protocol faithfully but passively intercepts client gradient updates to infer private training data.
- **Attacker Knowledge:**
  - Initial model architecture and parameters (`LeNet`, `models/global_model_init.pth`).
  - Loss function (Cross-Entropy).
  - Client gradient tensors (`updates/clientX_gradient.pt`).
- **Attacker Does NOT Know:**
  - Client private images (`clientX_ground_truth.pt` is reserved solely for quantitative scoring and visual evaluation).

---

## Methodology

### 1. Mathematical Formulation

Given target model parameters $W$ and received client gradient $\nabla W$, the attacker initializes a dummy image $x' \sim \mathcal{N}(0, 1)$ and dummy label logits $y'$.

The attacker minimizes the gradient matching distance:

$$\mathcal{D}(x', y') = \|\nabla W' - \nabla W\|_2^2 = \sum_{l=1}^L \|\nabla W'_l - \nabla W_l\|_2^2$$

where:
$$\nabla W' = \frac{\partial \mathcal{L}(f(x'; W), y')}{\partial W}$$

### 2. Optimization

- **Optimizer:** L-BFGS with line search, which rapidly navigates second-order curvature for smooth activation networks (such as `Tanh LeNet`).
- **Label Recovery:**
  - *Joint optimization (`joint`):* Optimizes continuous label logits concurrently with the dummy input.
  - *Analytical recovery (`analytical` / iDLG):* Exploits the property that for Cross-Entropy loss with softmax, $\frac{\partial \mathcal{L}}{\partial b_{fc}} = p - y$. The ground-truth class $c$ satisfies $\frac{\partial \mathcal{L}}{\partial b_c} = p_c - 1 < 0$, while all other classes $j \neq c$ have positive gradients ($p_j > 0$). Thus, $c = \arg\min_j \frac{\partial \mathcal{L}}{\partial b_j}$.
- **Image Bounding:** Images are clamped to $[0, 1]$ to match MNIST pixel values.

---

## Files

| File | Purpose |
|---|---|
| `dlg_attack.py` | `DLGAttacker` class: core gradient matching algorithm, optimizer closure, and snapshot tracking |
| `metrics.py` | Quantitative metrics: MSE, PSNR, SSIM, label accuracy, and attack success predicate |
| `reconstruction.py` | Driver script: loads gradients, executes attacks, generates comparison figures, logs metrics |
| `results/` | Output directory containing reconstructed images, progression plots, loss curves, and summary CSV/JSON |

---

## Quick Start

Run the attack from the repository root:

```bash
python task2_reconstruction_attack/reconstruction.py
```

### CLI Options

```bash
# Attack specific clients
python task2_reconstruction_attack/reconstruction.py --clients 1 2

# Change optimizer or iterations
python task2_reconstruction_attack/reconstruction.py --optimizer lbfgs --max-iter 100 --lr 1.0

# Use analytical label inference (iDLG mode)
python task2_reconstruction_attack/reconstruction.py --label-strategy analytical

# Use cosine similarity loss
python task2_reconstruction_attack/reconstruction.py --cost-fn cosine
```

---

## Results & Findings

### Quantitative Performance

Evaluated on all 4 clients from Task 1 (initial global model `global_model_init.pth`):

| Client ID | True Label | Recovered Label | Label Match | MSE | PSNR (dB) | SSIM | Attack Success | Iterations to Converge | Time (s) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Client 1** | 9 | 9 | 100% | $1.40 \times 10^{-7}$ | 68.55 dB | 0.9999 | **Success** | 12 | 0.34s |
| **Client 2** | 5 | 5 | 100% | $4.23 \times 10^{-8}$ | 73.73 dB | 1.0000 | **Success** | 13 | 0.34s |
| **Client 3** | 7 | 7 | 100% | $1.78 \times 10^{-7}$ | 67.49 dB | 0.9999 | **Success** | 12 | 0.30s |
| **Client 4** | 0 | 0 | 100% | $2.07 \times 10^{-7}$ | 66.83 dB | 0.9999 | **Success** | 13 | 0.33s |

- **Success Criteria:** Defined as $\text{MSE} \le 0.01$ and $\text{PSNR} \ge 20.0\text{ dB}$.
- **Average PSNR:** ~69.15 dB.
- **Average Execution Time:** ~0.33s per client on CPU.

### Generated Artifacts in `results/`

- `all_clients_summary.png`: Side-by-side grid comparing all 4 clients against ground truth.
- `clientX_comparison.png`: Dual-panel comparison (Ground Truth vs Reconstructed).
- `clientX_progression.png`: Visual progression across optimization steps (0, 5, 10, 20...).
- `clientX_loss_curve.png`: Log-scale plot of gradient distance minimization over iterations.
- `reconstruction_summary.csv` and `reconstruction_summary.json`: Complete quantitative evaluation records.

---

## Key Observations

1. **Severe Gradient Leakage:** In the absence of defense mechanisms, raw gradients completely leak both the image pixels and the true class label.
2. **Rapid Convergence:** Because the `LeNet` model utilizes smooth `Tanh` activations, L-BFGS converges in just 12–13 iterations to an MSE under $10^{-7}$.
3. **Necessity of Defenses:** These baseline results establish the critical need for gradient clipping and differential privacy noise, evaluated in Task 3 and Task 4.
