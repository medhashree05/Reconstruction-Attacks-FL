# Reconstruction Attacks in Federated Learning: Privacy Risks and Defense Mechanisms

A four-task Cryptography and Network Security project that studies **gradient-based data reconstruction in Federated Learning (FL)**. The project builds a small FL system using MNIST and PyTorch, demonstrates a Deep Leakage from Gradients (DLG)-style attack, applies gradient clipping and Gaussian-noise perturbation, and compares reconstruction quality and model utility.

> **Research question:** To what extent can a private MNIST image be reconstructed from a client gradient, and how do gradient clipping and Gaussian-noise perturbation affect reconstruction quality and model accuracy?

## Project at a glance

| Task | Purpose | Main outputs |
|---|---|---|
| **1. Federated Learning setup** | Train a global model with four clients and FedAvg; save single-image gradients for the attack | Initial/final model checkpoints, per-client gradients and updates, baseline accuracy |
| **2. Reconstruction attack** | Use gradient matching to infer a private image from a saved gradient | Original/reconstructed image comparisons, attack progression and loss plots, CSV/JSON metrics |
| **3. Defenses** | Clip gradients and generate clipped-plus-noisy gradient variants | Protected `.pt` gradient files and validation results |
| **4. Evaluation** | Run the same reconstruction attack against unprotected and protected gradients; separately compare model accuracy under update-level defenses | Per-experiment results, comparison tables, and plots |

## How the pipeline works

```text
Task 1: Federated Learning
MNIST -> 4 clients -> local training -> FedAvg
                         |
                         +-> single-sample gradients saved
                                      |
                                      v
Task 2: DLG-style reconstruction
Saved gradient + known model architecture/weights
              -> optimize dummy image to match gradient
              -> compare reconstruction with held-out ground truth
                                      |
                                      v
Task 3: Defense generation
Original gradient -> global-norm clipping
                  -> clipping + Gaussian noise
                                      |
                                      v
Task 4: Evaluation
Run the same attack against each gradient variant
+ train separate FL runs to compare final model accuracy
```

The project deliberately saves ground-truth images and labels separately. They are used only for scoring and visualization; the attack implementation should not use them as attack inputs.

## Repository structure

```text
Reconstruction-Attacks-FL-main/
├── README.md
├── requirements.txt
├── task1_fl_setup/
│   ├── dataset.py                  # MNIST download, client split, test loader
│   ├── model.py                    # LeNet-style CNN with Sigmoid activations
│   ├── client.py                   # local training and single-sample gradients
│   ├── server.py                   # FedAvg aggregation and model evaluation
│   ├── train.py                    # Task 1 entry point
│   ├── THREAT_MODEL.md             # attacker assumptions
│   ├── models/
│   │   ├── global_model_init.pth
│   │   ├── global_model.pth
│   │   └── baseline_results.json
│   └── updates/
│       ├── client*_gradient.pt
│       ├── client*_ground_truth.pt
│       └── client*_update.pt
├── task2_reconstruction_attack/
│   ├── dlg_attack.py               # DLG-style gradient-matching optimizer
│   ├── metrics.py                  # MSE, PSNR, SSIM, label match, success
│   ├── reconstruction.py            # attack runner and visualizations
│   ├── validate_task2.py            # Task 2 checks
│   └── results/                    # images, plots, CSV/JSON summaries
├── task3_defenses/
│   ├── gradient_clipping.py         # global L2 norm clipping
│   ├── differential_privacy.py      # clipping plus Gaussian noise
│   ├── evaluate_defenses.py         # attack evaluation on protected gradients
│   ├── analyze_results.py
│   ├── validate_task3.py
│   ├── protected_updates/           # generated protected gradient tensors
│   └── results/                    # validation/evaluation outputs
└── task4_evaluation/
    ├── experiments.py               # run attacks across gradient variants
    ├── comparison.py                # combine reconstruction metrics
    ├── plots.py                     # reconstruction comparison plots
    ├── accuracy_evaluation.py       # train FL under update-level defenses
    ├── accuracy_plots.py
    └── results/                     # experiment summaries and graphs
```

## Requirements

- Python 3.10+ recommended
- PyTorch
- torchvision
- NumPy
- Matplotlib
- pandas
- scikit-image
- tqdm

A CUDA-compatible GPU is optional; the experiments can run on CPU, although attack runs may take longer. The first Task 1 run downloads MNIST through `torchvision`.

## Installation

Open a terminal at the repository root (the directory containing `README.md`).

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, you can run the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the project

Run commands from the **repository root** unless stated otherwise. Run tasks in order so that each task has the files it needs.

### Task 1 — Build the FL baseline and save gradients

```bash
python task1_fl_setup/train.py
```

The script creates four clients, trains with FedAvg, prints test accuracy per round, and writes:
- `task1_fl_setup/models/global_model_init.pth` — model before federated training; used by the reconstruction experiment.
- `task1_fl_setup/models/global_model.pth` — model after the final FL round.
- `task1_fl_setup/models/baseline_results.json` — initial and per-round test accuracy plus configuration.
- `task1_fl_setup/updates/clientN_gradient.pt` — one private sample's gradient for each client, computed at the initial model.
- `task1_fl_setup/updates/clientN_ground_truth.pt` — corresponding image/label for evaluation only.
- `task1_fl_setup/updates/clientN_update.pt` — last-round client model delta.

The checked-in baseline results record **96.48% final test accuracy after 20 rounds**, with four clients, two local epochs, and learning rate 0.01. Results can vary with environment or code/configuration changes; the JSON generated by your own run is the authoritative result for that run.

### Task 2 — Run the reconstruction attack

```bash
python task2_reconstruction_attack/reconstruction.py
```

Useful options:

```bash
# Attack selected clients
python task2_reconstruction_attack/reconstruction.py --clients 1 2

# Set optimization iterations and learning rate
python task2_reconstruction_attack/reconstruction.py --max-iter 100 --lr 1.0

# Select optimizer
python task2_reconstruction_attack/reconstruction.py --optimizer lbfgs
python task2_reconstruction_attack/reconstruction.py --optimizer adam

# Use analytical label inference or cosine distance
python task2_reconstruction_attack/reconstruction.py --label-strategy analytical
python task2_reconstruction_attack/reconstruction.py --cost-fn cosine
```

Check `python task2_reconstruction_attack/reconstruction.py --help` for the exact options supported by your local version. Outputs are saved in `task2_reconstruction_attack/results/`, including per-client comparisons, reconstruction progression images, gradient-matching loss curves, and summary CSV/JSON files.

### Task 3 — Generate protected gradients

Run both scripts from the repository root:

```bash
python task3_defenses/gradient_clipping.py
python task3_defenses/differential_privacy.py
```

Gradient clipping evaluates global L2 thresholds **1.0, 0.5, and 0.1**. The Gaussian-noise script first clips to norm 1.0 and then adds noise with levels **0, 0.01, 0.05, 0.1, and 0.5**.

Protected gradient tensors are written to `task3_defenses/protected_updates/`. Running the scripts again overwrites files with matching names.

### Task 4 — Compare reconstruction and model utility

Run the reconstruction experiments and combine the results:

```bash
python task4_evaluation/experiments.py
python task4_evaluation/comparison.py
python task4_evaluation/plots.py
```

To evaluate **global model accuracy** when clipping/noise is applied to client model updates during FL training, run:

```bash
python task4_evaluation/accuracy_evaluation.py
python task4_evaluation/accuracy_plots.py
```

These are two related but distinct experiments:
1. **Gradient reconstruction experiment:** protects the single-sample gradients saved by Task 1 and runs the attack against them.
2. **Model utility experiment:** performs new FL training runs and protects each client's model delta before aggregation.

Do not treat the two as the exact same mechanism or assume their parameter values have identical meaning.

## Methodology

### Task 1: Federated Learning

- Dataset: MNIST, 28 × 28 grayscale images with ten classes.
- Client count: four.
- Client partition: disjoint IID subsets; each client receives 2,000 training examples in the current implementation.
- Model: a small LeNet-style CNN with Sigmoid activations.
- Local training: SGD.
- Aggregation: FedAvg, implemented by averaging client model deltas and adding the average delta to the global parameters.
- Gradient target: the gradient of cross-entropy loss for one client image, computed using the **initial global model**.

### Task 2: DLG-style reconstruction

The attacker starts with a dummy image and seeks to make its model gradient match the observed target gradient. It iteratively minimizes a gradient-distance objective, then compares the inferred image with the ground-truth image outside the attack loop.

This is a controlled demonstration using a single image per client. It does not imply that every FL gradient or batch can be reconstructed equally well.

### Task 3: Defenses

**Global-norm gradient clipping** rescales the concatenated gradient collection if its L2 norm exceeds a threshold:

`scale = min(1, C / (||g|| + epsilon))`

**Gaussian-noise perturbation** clips to a norm of 1.0 and adds random Gaussian noise to each gradient tensor:

`protected_gradient = clipped_gradient + noise`

**Privacy terminology caution:** the code demonstrates clipping and Gaussian-noise perturbation, but the configured noise values are not accompanied by a formal privacy accountant or a calibrated `(epsilon, delta)` guarantee. Therefore, results should be described as *Gaussian-noise perturbation / DP-inspired protection*, not as a formally proven differential-privacy guarantee.

### Task 4: Evaluation metrics

- **MSE:** mean squared pixel error; lower means closer reconstruction.
- **PSNR:** peak signal-to-noise ratio in dB; higher generally means closer reconstruction.
- **SSIM:** structural similarity; higher generally means more similar structure.
- **Label match:** whether the classifier's predicted label for the reconstructed image matches the ground-truth label.
- **Attack success:** the success flag produced by the project's metric function and its configured threshold. Check `task2_reconstruction_attack/metrics.py` before interpreting this as a universal criterion.
- **Model accuracy:** test accuracy of the global model; higher indicates better utility.

## Results included in this repository

The repository contains saved experiment outputs. The following are **the values recorded in those files**, not guarantees for future runs. Reconstruction results are averaged over four client images.

### Reconstruction attack results

| Gradient variant | Mean MSE ↓ | Mean PSNR (dB) ↑ | Mean SSIM ↑ | Attack success |
|---|---:|---:|---:|---:|
| No defense | 0.000000190 | 67.33 | 0.99994 | 4/4 |
| Clipping, C = 1.0 | 0.000000459 | 63.46 | 0.99981 | 4/4 |
| Clipping, C = 0.5 | 0.00000608 | 53.25 | 0.99780 | 4/4 |
| Clipping, C = 0.1 | 0.000398 | 34.21 | 0.94949 | 4/4 |
| Noise = 0.01 | 0.05762 | 12.42 | 0.57960 | 0/4 |
| Noise = 0.05 | 0.42400 | 3.73 | 0.09461 | 0/4 |
| Noise = 0.1 | 0.46884 | 3.29 | 0.01809 | 0/4 |
| Noise = 0.5 | 0.48059 | 3.18 | 0.01253 | 0/4 |

`Noise = 0` is a control case in the DP-gradient experiment, not an additional privacy protection level.

### Model utility results

The saved `task4_evaluation/results/model_accuracy_results.csv` records these final test accuracies:

| Experiment | Final accuracy |
|---|---:|
| No defense | 96.48% |
| Clip update, C = 1.0 | 96.42% |
| Clip update, C = 0.5 | 95.44% |
| Clip update, C = 0.1 | 88.67% |
| Noise on update, 0.01 | 96.32% |
| Noise on update, 0.05 | 95.63% |
| Noise on update, 0.1 | 90.55% |
| Noise on update, 0.5 | 11.57% |

### Interpretation

- The unprotected single-image gradients in the recorded experiment allowed highly accurate reconstructions.
- Clipping alone reduced reconstruction quality as the threshold became smaller, but did not prevent the configured attack from succeeding on these four samples.
- Gaussian noise at the tested levels substantially degraded the recorded reconstructions.
- In the separate FL utility experiment, stronger perturbation reduced final model accuracy, particularly at noise level 0.5.
- These are small, controlled experiments on four MNIST samples. They are useful demonstrations, not broad claims about all FL systems.

## Important limitations

1. **Small experiment:** MNIST and four clients are chosen for a manageable demonstration; results may not generalize to complex datasets or larger models.
2. **Single-sample gradients:** the reconstruction target is a gradient from one image. Batch reconstruction can behave differently.
3. **Ground-truth access:** ground-truth images are saved for evaluation and must not be passed into the attack optimizer.
4. **DP guarantee:** adding Gaussian noise alone does not establish formal differential privacy. A defensible DP claim requires a correctly specified clipping/sensitivity setup, sampling assumptions, and privacy accounting.
5. **Secure aggregation:** it is not implemented as a primary defense in this repository; it may be discussed as complementary work.
6. **Reproducibility:** generated `.pt`, `.png`, `.csv`, and `.json` files can be overwritten by rerunning scripts. Keep copies of results used in a report.

## References

1. Zhu, L., Liu, Z., and Han, S. “Deep Leakage from Gradients.” *Advances in Neural Information Processing Systems (NeurIPS)*, 2019.
2. Zhao, B., Mopuri, K. R., and Bilen, H. “iDLG: Improved Deep Leakage from Gradients,” 2020.
3. McMahan, B. et al. “Communication-Efficient Learning of Deep Networks from Decentralized Data.” *AISTATS*, 2017.
4. Geiping, J. et al. “Inverting Gradients—How Easy Is It to Break Privacy in Federated Learning?” *NeurIPS*, 2020.
5. Abadi, M. et al. “Deep Learning with Differential Privacy.” *ACM CCS*, 2016.
6. Bonawitz, K. et al. “Practical Secure Aggregation for Privacy-Preserving Machine Learning.” *ACM CCS*, 2017.
