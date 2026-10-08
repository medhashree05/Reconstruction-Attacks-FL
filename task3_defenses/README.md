# Task 3 - Defenses Against Gradient Reconstruction Attacks

## 1. Overview

Task 3 evaluates defenses against gradient reconstruction attacks in Federated Learning (FL).

The defenses implemented are:

1. Gradient Clipping
2. Gaussian Noise Perturbation for Differential Privacy (DP)

The protected gradients are evaluated using the same Deep Leakage from Gradients (DLG) reconstruction attack implemented in Task 2.

The goal is to determine how effectively each defense reduces the ability of an attacker to reconstruct private training images from shared gradients.

---

## 2. Input Data

Task 3 uses the gradient files generated during Task 1.

Input directory:

task1_fl_setup/updates/

The relevant gradient files are:

client1_gradient.pt
client2_gradient.pt
client3_gradient.pt
client4_gradient.pt

The corresponding ground-truth files are:

client1_ground_truth.pt
client2_ground_truth.pt
client3_ground_truth.pt
client4_ground_truth.pt

The ground-truth files are used ONLY for evaluating the quality of the reconstructed images and labels.

They are NOT provided as input to the reconstruction attack.

The gradient files contain lists of tensors corresponding to the gradients of the LeNet model parameters.

---

## 3. Gradient Clipping

### 3.1 Purpose

Gradient clipping limits the magnitude of the gradient vector before it is shared.

This limits the sensitivity of the gradients and prevents unusually large gradients from being directly exposed.

### 3.2 Implementation

The global L2 norm of all gradient tensors is calculated.

If the gradient norm exceeds the clipping threshold, all gradient tensors are scaled by the same factor.

The clipping scale is:

scale = min(1, C / ||g||)

where:

C  = clipping threshold

||g|| = global L2 norm of the gradient

### 3.3 Clipping Values

The following clipping thresholds were evaluated:

C = 1.0
C = 0.5
C = 0.1

### 3.4 Implementation File

task3_defenses/gradient_clipping.py

### 3.5 Protected Files

The protected gradients are stored in:

task3_defenses/protected_updates/

Example files:

client1_gradient_clip_1.0.pt
client1_gradient_clip_0.5.pt
client1_gradient_clip_0.1.pt

The same files are generated for clients 2, 3, and 4.

---

## 4. Differential Privacy / Gaussian Noise

### 4.1 Purpose

Gaussian noise is added to clipped gradients to make gradient reconstruction more difficult.

The implementation first clips the gradient and then adds Gaussian noise.

For each gradient tensor:

noisy_gradient = clipped_gradient + Gaussian_noise

The Gaussian noise is generated using:

torch.randn_like(gradient) * noise_level

### 4.2 Noise Levels

The following noise levels were evaluated:

0
0.01
0.05
0.1
0.5

A noise level of 0 acts as a control case corresponding to clipping without additional noise.

### 4.3 Implementation File

task3_defenses/differential_privacy.py

### 4.4 Protected Files

Example files:

client1_gradient_dp_0.0.pt
client1_gradient_dp_0.01.pt
client1_gradient_dp_0.05.pt
client1_gradient_dp_0.1.pt
client1_gradient_dp_0.5.pt

The same files are generated for clients 2, 3, and 4.

---

## 5. Defense Validation

The generated protected gradients were first validated independently.

### 5.1 Gradient Clipping

Validation results confirm that the protected gradient norm is approximately equal to the selected clipping threshold.

Examples:

C = 1.0  -> Protected norm approximately 1.0

C = 0.5  -> Protected norm approximately 0.5

C = 0.1  -> Protected norm approximately 0.1

The complete validation results are stored in:

task3_defenses/results/clipping_results.csv

### 5.2 Differential Privacy

The Gaussian-noise validation results are stored in:

task3_defenses/results/dp_results.csv

---

## 6. Reconstruction Attack Evaluation

After applying each defense, the protected gradients were given to the same Deep Leakage from Gradients (DLG) reconstruction attack implemented in Task 2.

This ensures that the defense evaluation uses the same attack methodology as the baseline experiment.

The attack configuration was:

Optimizer       : LBFGS
Learning rate   : 1.0
Maximum steps   : 100
Cost function   : L2
Label strategy  : Joint
Device          : CPU
Tolerance       : 1e-6
Seed            : 42
Image shape     : 1 x 1 x 28 x 28

The ground-truth images and labels were used only after reconstruction to calculate evaluation metrics.

The protected gradients themselves are the only gradient inputs provided to the attacker.

---

## 7. Evaluation Metrics

The following metrics were used.

### 7.1 Mean Squared Error (MSE)

MSE measures the pixel-wise difference between the reconstructed image and the original image.

Lower MSE indicates a more accurate reconstruction.

### 7.2 Peak Signal-to-Noise Ratio (PSNR)

PSNR measures similarity between the reconstructed image and the original image.

Higher PSNR indicates better reconstruction.

### 7.3 Structural Similarity Index (SSIM)

SSIM measures structural similarity between the reconstructed image and the original image.

Values closer to 1 indicate stronger similarity.

### 7.4 Attack Success

Attack success indicates whether the reconstruction satisfies the success criteria used by the Task 2 evaluation.

### 7.5 Label Recovery

Label recovery measures whether the attacker correctly recovered the original class label.

---

## 8. Experimental Setup

A total of 32 reconstruction experiments were performed.

### Gradient Clipping

4 clients x 3 clipping values = 12 experiments

### Gaussian Noise

4 clients x 5 noise levels = 20 experiments

### Total

12 + 20 = 32 experiments

Each protected gradient was evaluated using the same DLG attack configuration.

---

## 9. Gradient Clipping Results

| Clipping C | Avg MSE | Avg PSNR (dB) | Avg SSIM | Attack Success | Label Recovery |
|------------|---------|---------------|----------|----------------|----------------|
| 1.0        | 0.00000096 | 61.8838 | 0.999583 | 100% | 100% |
| 0.5        | 0.00000764 | 52.0445 | 0.997556 | 100% | 100% |
| 0.1        | 0.00030729 | 35.2236 | 0.958157 | 100% | 100% |

### Observation

Reducing the clipping threshold progressively reduced reconstruction quality.

Average PSNR:

C = 1.0 -> 61.88 dB

C = 0.5 -> 52.04 dB

C = 0.1 -> 35.22 dB

Average SSIM:

C = 1.0 -> 0.999583

C = 0.5 -> 0.997556

C = 0.1 -> 0.958157

However, the attack success rate remained 100% for all tested clipping thresholds.

Therefore, gradient clipping alone degraded reconstruction quality but did not prevent successful DLG reconstruction under the tested conditions.

---

## 10. Differential Privacy / Gaussian Noise Results

| Noise sigma | Avg MSE | Avg PSNR (dB) | Avg SSIM | Attack Success | Label Recovery |
|-------------|---------|---------------|----------|----------------|----------------|
| 0.00        | 0.00000096 | 61.8838 | 0.999583 | 100% | 100% |
| 0.01        | 0.05762540 | 12.4182 | 0.579568 | 0% | 100% |
| 0.05        | 0.41176780 | 3.8648 | 0.117166 | 0% | 100% |
| 0.10        | 0.46770270 | 3.3045 | 0.041406 | 0% | 50% |
| 0.50        | 0.47665440 | 3.2204 | 0.012963 | 0% | 25% |

### Observation

Adding Gaussian noise substantially reduced reconstruction quality.

At the smallest non-zero tested noise level:

sigma = 0.01

the average results were:

MSE  = 0.05762540

PSNR = 12.4182 dB

SSIM = 0.579568

The attack success rate decreased from:

100% -> 0%

Increasing the noise level further reduced reconstruction quality.

At:

sigma = 0.50

the average results were:

MSE  = 0.47665440

PSNR = 3.2204 dB

SSIM = 0.012963

The attack success rate remained:

0%

Label recovery also decreased at higher noise levels.

---

## 11. Overall Experimental Results

Across all 32 experiments:

Average MSE       : 0.17675838

Average PSNR      : 29.2305 dB

Average SSIM      : 0.588248

Attack Success    : 50.00%

Label Recovery    : 84.38%

The overall attack success rate is 50% because:

12/12 clipping experiments       -> successful

4/4 zero-noise DP experiments    -> successful

0/16 non-zero-noise DP attacks   -> successful

Therefore:

16 successful attacks / 32 experiments = 50%

---

## 12. Main Findings

### Finding 1 - Gradient Clipping Alone Is Insufficient

Gradient clipping reduced reconstruction quality as the clipping threshold became smaller.

However, all tested clipping levels still allowed successful reconstruction.

Therefore:

Gradient clipping
        |
        v
Reduces gradient magnitude
        |
        v
Reduces reconstruction quality
        |
        v
But does not prevent DLG reconstruction

under the tested conditions.

### Finding 2 - Gaussian Noise Provides Stronger Protection

Adding Gaussian noise substantially reduced reconstruction quality.

All tested non-zero noise levels resulted in:

Attack success = 0%

The smallest tested non-zero noise level:

sigma = 0.01

was already sufficient to prevent successful reconstruction under the tested DLG configuration.

### Finding 3 - Higher Noise Reduces Image Similarity

Increasing the noise level caused:

Higher MSE
Lower PSNR
Lower SSIM

This demonstrates a privacy-utility trade-off.

Higher noise provides stronger protection against reconstruction but also causes greater distortion of the gradient information.

### Finding 4 - Label Leakage Can Persist

Preventing image reconstruction does not necessarily prevent all information leakage.

For example:

sigma = 0.01 -> Label recovery = 100%

sigma = 0.05 -> Label recovery = 100%

sigma = 0.10 -> Label recovery = 50%

sigma = 0.50 -> Label recovery = 25%

Therefore, although image reconstruction was unsuccessful at non-zero noise levels, the attacker could still recover labels at some noise levels.

---

## 13. Defense Comparison

| Defense | Parameter | Reconstruction Quality | Attack Success | Label Recovery |
|---------|-----------|------------------------|----------------|----------------|
| Clipping | C = 1.0 | Very high | 100% | 100% |
| Clipping | C = 0.5 | Very high | 100% | 100% |
| Clipping | C = 0.1 | Reduced | 100% | 100% |
| Gaussian Noise | sigma = 0.01 | Strongly reduced | 0% | 100% |
| Gaussian Noise | sigma = 0.05 | Very strongly reduced | 0% | 100% |
| Gaussian Noise | sigma = 0.10 | Very strongly reduced | 0% | 50% |
| Gaussian Noise | sigma = 0.50 | Extremely low | 0% | 25% |

The results indicate that Gaussian noise was substantially more effective than gradient clipping alone against the tested gradient reconstruction attack.

---

## 14. Output Files

The main reconstruction evaluation results are stored in:

task3_defenses/results/reconstruction_defense_results.csv

task3_defenses/results/reconstruction_defense_results.json

Summary files:

task3_defenses/results/clipping_reconstruction_summary.csv

task3_defenses/results/dp_reconstruction_summary.csv

task3_defenses/results/defense_comparison_summary.csv

Defense validation files:

task3_defenses/results/clipping_results.csv

task3_defenses/results/dp_results.csv

---

## 15. Directory Structure

task3_defenses/
|
+-- gradient_clipping.py
+-- differential_privacy.py
+-- validate_task3.py
+-- evaluate_defenses.py
+-- analyze_results.py
+-- README.md
|
+-- protected_updates/
|   +-- client1_gradient_clip_1.0.pt
|   +-- client1_gradient_clip_0.5.pt
|   +-- client1_gradient_clip_0.1.pt
|   +-- client1_gradient_dp_0.0.pt
|   +-- client1_gradient_dp_0.01.pt
|   +-- client1_gradient_dp_0.05.pt
|   +-- client1_gradient_dp_0.1.pt
|   +-- client1_gradient_dp_0.5.pt
|   +-- ...
|
+-- results/
    +-- clipping_results.csv
    +-- dp_results.csv
    +-- reconstruction_defense_results.csv
    +-- reconstruction_defense_results.json
    +-- clipping_reconstruction_summary.csv
    +-- dp_reconstruction_summary.csv
    +-- defense_comparison_summary.csv

---

## 16. How to Run

All commands should be executed from the repository root.

### 16.1 Generate Gradient Clipping Results

python task3_defenses/gradient_clipping.py

This generates protected gradients for:

C = 1.0
C = 0.5
C = 0.1

### 16.2 Generate Gaussian-Noise Protected Gradients

python task3_defenses/differential_privacy.py

This generates protected gradients for:

sigma = 0
sigma = 0.01
sigma = 0.05
sigma = 0.1
sigma = 0.5

### 16.3 Validate the Generated Defenses

python task3_defenses/validate_task3.py

This generates:

clipping_results.csv

dp_results.csv

### 16.4 Run Reconstruction Evaluation

The reconstruction evaluation reuses the DLG attack implementation from Task 2.

Run:

python -m task3_defenses.evaluate_defenses

This evaluates all protected gradients and generates:

reconstruction_defense_results.csv

reconstruction_defense_results.json

### 16.5 Analyze Reconstruction Results

Run:

python -m task3_defenses.analyze_results

This generates:

clipping_reconstruction_summary.csv

dp_reconstruction_summary.csv

defense_comparison_summary.csv

---

## 17. Relationship With Previous Tasks

### Task 1

Task 1 generated the federated learning model, client gradients, client updates, and ground-truth samples.

Task 3 uses:

client*_gradient.pt

as the raw gradient input.

The ground-truth files are used only for post-attack evaluation.

### Task 2

Task 2 implemented the Deep Leakage from Gradients (DLG) reconstruction attack.

Task 3 reuses the same DLG implementation against the protected gradients.

The workflow is:

Task 1
   |
   v
Client gradients
   |
   v
Task 3 defense
   |
   v
Protected gradients
   |
   v
Task 2 DLG attack
   |
   v
Reconstructed image
   |
   v
MSE / PSNR / SSIM / Label Recovery

This provides a consistent comparison between undefended and defended gradients.

---

## 18. Important Privacy Note

The Gaussian-noise implementation in this task is an experimental privacy perturbation using the specified noise levels.

The experiment does NOT calculate formal differential privacy guarantees in terms of:

epsilon (epsilon)

delta (delta)

Therefore, these experimental results should not be interpreted as a formal (epsilon, delta)-differential privacy guarantee.

A formal DP analysis would require additional assumptions and privacy accounting based on factors such as:

- Mechanism sensitivity
- Sampling procedure
- Number of compositions
- Clipping bounds
- Noise calibration
- Privacy accountant

The current implementation is intended to experimentally evaluate the effect of Gaussian noise on gradient reconstruction attacks.

---

## 19. Reproducibility

The reconstruction attack uses:

Seed = 42

The protected gradient files used for evaluation are saved in:

task3_defenses/protected_updates/

Therefore, the current reconstruction evaluation operates on the saved protected-gradient artifacts.

The Gaussian-noise generation uses random Gaussian noise. Regenerating the protected DP files may produce different noise samples unless the random seed is explicitly controlled during generation.

---

## 20. Final Summary

The Task 3 experiments demonstrate the following:

- Gradient clipping reduces the quality of reconstructed images as the clipping threshold decreases.
- Gradient clipping alone did not prevent successful DLG reconstruction for the tested thresholds.
- Gaussian noise substantially degraded reconstructed image quality.
- Every tested non-zero Gaussian noise level resulted in 0% attack success.
- Higher Gaussian noise levels resulted in progressively lower PSNR and SSIM.
- Label recovery remained possible at lower noise levels even when image reconstruction failed.
- The strongest tested noise level, sigma = 0.5, reduced average SSIM to approximately 0.013 and maintained a 0% reconstruction attack success rate.
- The experiments demonstrate a clear privacy-utility trade-off when protecting gradients with Gaussian noise.

Overall, under the tested experimental configuration, Gaussian noise provided substantially stronger protection against gradient reconstruction attacks than gradient clipping alone.