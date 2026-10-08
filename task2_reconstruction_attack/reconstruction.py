"""Driver script to run DLG reconstruction attack on saved client gradients.

Run from repository root:
    python task2_reconstruction_attack/reconstruction.py
"""

import argparse
import json
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import torch

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from task1_fl_setup.model import LeNet
from task2_reconstruction_attack.dlg_attack import DLGAttacker
from task2_reconstruction_attack.metrics import evaluate_reconstruction


def save_single_image(tensor, path):
    """Save 1x1x28x28 or 28x28 tensor as grayscale PNG."""
    arr = tensor.detach().cpu().squeeze().numpy()
    plt.figure(figsize=(2.5, 2.5))
    plt.imshow(arr, cmap="gray", vmin=0.0, vmax=1.0)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_comparison_plot(true_img, recon_img, metrics, cid, path):
    """Save side-by-side comparison plot of ground truth vs reconstruction."""
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    
    true_arr = true_img.detach().cpu().squeeze().numpy()
    recon_arr = recon_img.detach().cpu().squeeze().numpy()
    
    axes[0].imshow(true_arr, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title(f"Client {cid}: Ground Truth\n(Label: {metrics.get('true_label', '?')})", fontsize=10)
    axes[0].axis("off")
    
    axes[1].imshow(recon_arr, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title(
        f"Reconstructed (DLG)\nPSNR: {metrics['psnr']:.2f} dB | MSE: {metrics['mse']:.5f}",
        fontsize=10,
    )
    axes[1].axis("off")
    
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def save_progression_plot(progression, cid, path):
    """Save multi-panel plot showing reconstruction evolution over iterations."""
    steps = sorted(progression.keys())
    n = len(steps)
    fig, axes = plt.subplots(1, n, figsize=(2.2 * n, 2.6))
    if n == 1:
        axes = [axes]
        
    for ax, step in zip(axes, steps):
        img_arr = progression[step].squeeze().numpy()
        ax.imshow(img_arr, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(f"Iter {step}", fontsize=10)
        ax.axis("off")
        
    plt.suptitle(f"Client {cid}: DLG Reconstruction Progression", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def save_loss_curve(loss_history, cid, path):
    """Save gradient distance loss curve over iterations."""
    plt.figure(figsize=(5, 3.5))
    plt.plot(range(1, len(loss_history) + 1), loss_history, color="#1f77b4", lw=1.8)
    plt.yscale("log")
    plt.xlabel("Iteration", fontsize=10)
    plt.ylabel("Gradient Distance (L2)", fontsize=10)
    plt.title(f"Client {cid}: Attack Convergence Loss", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def save_summary_grid(client_results, path):
    """Save summary grid of all clients (Ground Truth vs Reconstructed)."""
    n = len(client_results)
    fig, axes = plt.subplots(2, n, figsize=(2.5 * n, 5.2))
    
    for i, res in enumerate(client_results):
        cid = res["client_id"]
        true_arr = res["true_image"].squeeze().numpy()
        recon_arr = res["recon_image"].squeeze().numpy()
        
        # Row 0: True image
        axes[0, i].imshow(true_arr, cmap="gray", vmin=0.0, vmax=1.0)
        axes[0, i].set_title(f"Client {cid} Ground Truth\nDigit: {res['true_label']}", fontsize=9)
        axes[0, i].axis("off")
        
        # Row 1: Reconstructed image
        axes[1, i].imshow(recon_arr, cmap="gray", vmin=0.0, vmax=1.0)
        axes[1, i].set_title(
            f"DLG Recon (Label: {res['pred_label']})\nPSNR: {res['psnr']:.1f} dB",
            fontsize=9,
        )
        axes[1, i].axis("off")
        
    plt.suptitle("DLG Reconstruction Attack Summary (Unprotected Gradients)", fontsize=13, y=0.98)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run Task 2 DLG Reconstruction Attack.")
    parser.add_argument("--clients", nargs="+", type=int, default=[1, 2, 3, 4],
                        help="Client IDs to attack (default: 1 2 3 4)")
    parser.add_argument("--max-iter", type=int, default=100,
                        help="Maximum optimization iterations (default: 100)")
    parser.add_argument("--lr", type=float, default=1.0,
                        help="Optimization learning rate (default: 1.0)")
    parser.add_argument("--optimizer", type=str, default="lbfgs", choices=["lbfgs", "adam"],
                        help="Optimizer type (default: lbfgs)")
    parser.add_argument("--cost-fn", type=str, default="l2", choices=["l2", "cosine"],
                        help="Gradient matching cost function (default: l2)")
    parser.add_argument("--label-strategy", type=str, default="joint", choices=["joint", "analytical", "known"],
                        help="Label restoration strategy (default: joint)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to run on (default: cpu)")
    parser.add_argument("--tolerance", type=float, default=1e-6,
                        help="Early stopping gradient difference tolerance (default: 1e-6)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--model-path", type=str,
                        default=os.path.join(REPO_ROOT, "task1_fl_setup", "models", "global_model_init.pth"),
                        help="Path to initial global model weights")
    parser.add_argument("--updates-dir", type=str,
                        default=os.path.join(REPO_ROOT, "task1_fl_setup", "updates"),
                        help="Directory containing client gradients and ground truths")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(REPO_ROOT, "task2_reconstruction_attack", "results"),
                        help="Directory to save attack results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 68)
    print(" TASK 2: DEEP LEAKAGE FROM GRADIENTS (DLG) RECONSTRUCTION ATTACK")
    print("=" * 68)
    print(f"Initial model:     {args.model_path}")
    print(f"Updates directory: {args.updates_dir}")
    print(f"Target clients:    {args.clients}")
    print(f"Optimizer:         {args.optimizer.upper()} (lr={args.lr})")
    print(f"Label strategy:    {args.label_strategy}")
    print(f"Cost function:     {args.cost_fn.upper()}")
    print(f"Max iterations:    {args.max_iter}")
    print(f"Device:            {args.device}")
    print("-" * 68)

    # 1. Load initial model
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    model = LeNet().to(args.device)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.eval()

    attacker = DLGAttacker(
        model=model,
        device=args.device,
        optimizer_type=args.optimizer,
        lr=args.lr,
        max_iter=args.max_iter,
        cost_fn=args.cost_fn,
        label_strategy=args.label_strategy,
        tolerance=args.tolerance,
        seed=args.seed,
    )

    client_records = []
    detailed_metrics = []

    for cid in args.clients:
        print(f"\n[+] Attacking Client {cid}...")
        grad_file = os.path.join(args.updates_dir, f"client{cid}_gradient.pt")
        gt_file = os.path.join(args.updates_dir, f"client{cid}_ground_truth.pt")

        if not os.path.exists(grad_file):
            print(f"[-] Gradient file {grad_file} not found. Skipping.")
            continue

        target_grads = torch.load(grad_file, map_location=args.device)

        # Load ground truth for evaluation only
        true_img = None
        true_label = None
        if os.path.exists(gt_file):
            gt = torch.load(gt_file, map_location=args.device)
            true_img = gt["image"]
            true_label = gt["label"]

        # Run attack
        attack_result = attacker.attack(
            target_gradients=target_grads,
            img_shape=(1, 1, 28, 28),
            true_image=true_img,
            true_label=true_label,
            log_interval=20,
            progression_steps=[0, 5, 10, 20, 50, args.max_iter],
        )

        recon_img = attack_result["reconstructed_image"]
        pred_label = attack_result["reconstructed_label"]

        # Evaluate metrics
        eval_dict = evaluate_reconstruction(
            recon_img=recon_img,
            true_img=true_img,
            pred_label=pred_label,
            true_label=true_label,
        )

        eval_dict["client_id"] = cid
        eval_dict["iterations"] = attack_result["iterations"]
        eval_dict["converged"] = attack_result["converged"]
        eval_dict["duration_sec"] = attack_result["duration"]
        eval_dict["final_grad_loss"] = (
            attack_result["loss_history"][-1] if attack_result["loss_history"] else None
        )

        detailed_metrics.append(eval_dict)

        # Save artifacts
        save_single_image(recon_img, os.path.join(args.output_dir, f"client{cid}_recon.png"))
        if cid == 1:
            save_single_image(recon_img, os.path.join(args.output_dir, "recon_nodefense.png"))
        if true_img is not None:
            save_single_image(true_img, os.path.join(args.output_dir, f"client{cid}_original.png"))
            if cid == 1:
                save_single_image(true_img, os.path.join(args.output_dir, "original.png"))
            save_comparison_plot(
                true_img,
                recon_img,
                eval_dict,
                cid,
                os.path.join(args.output_dir, f"client{cid}_comparison.png"),
            )
            # Stash for summary grid
            client_records.append({
                "client_id": cid,
                "true_image": true_img.cpu(),
                "recon_image": recon_img.cpu(),
                "true_label": eval_dict.get("true_label", "?"),
                "pred_label": eval_dict.get("pred_label", "?"),
                "psnr": eval_dict["psnr"],
                "mse": eval_dict["mse"],
            })

        save_progression_plot(
            attack_result["progression"],
            cid,
            os.path.join(args.output_dir, f"client{cid}_progression.png"),
        )
        save_loss_curve(
            attack_result["loss_history"],
            cid,
            os.path.join(args.output_dir, f"client{cid}_loss_curve.png"),
        )

        print(
            f"    -> Client {cid} Finished: Label={eval_dict.get('pred_label')} "
            f"(GT={eval_dict.get('true_label')}, Match={eval_dict.get('label_match')}) | "
            f"MSE={eval_dict['mse']:.6f} | PSNR={eval_dict['psnr']:.2f} dB | "
            f"SSIM={eval_dict['ssim']:.4f} | Converged in {attack_result['iterations']} iters ({attack_result['duration']:.2f}s)"
        )

    # Save summary grid if ground truths are available
    if client_records:
        save_summary_grid(client_records, os.path.join(args.output_dir, "all_clients_summary.png"))

    # Save CSV and JSON summaries
    df = pd.DataFrame(detailed_metrics)
    csv_path = os.path.join(args.output_dir, "reconstruction_summary.csv")
    json_path = os.path.join(args.output_dir, "reconstruction_summary.json")

    # Clean DataFrame column ordering
    col_order = [
        "client_id",
        "true_label",
        "pred_label",
        "label_match",
        "mse",
        "psnr",
        "ssim",
        "success",
        "iterations",
        "final_grad_loss",
        "duration_sec",
    ]
    cols = [c for c in col_order if c in df.columns] + [c for c in df.columns if c not in col_order]
    df = df[cols]
    df.to_csv(csv_path, index=False)

    with open(json_path, "w") as f:
        json.dump(detailed_metrics, f, indent=2)

    print("\n" + "=" * 68)
    print(" RECONSTRUCTION ATTACK SUMMARY RESULTS")
    print("=" * 68)
    print(df.to_string(index=False))
    print("-" * 68)
    print(f"Results saved to: {args.output_dir}/")
    print(f"Summary CSV:      {csv_path}")
    print(f"Summary JSON:     {json_path}")
    print("=" * 68)


if __name__ == "__main__":
    main()
