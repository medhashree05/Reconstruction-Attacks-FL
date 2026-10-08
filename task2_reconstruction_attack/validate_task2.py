"""Validation script for Task 2 Reconstruction Attack outputs."""

import json
import os
import sys
import pandas as pd
from PIL import Image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(REPO_ROOT, "task2_reconstruction_attack", "results")
CLIENTS = [1, 2, 3, 4]


def main():
    print("=" * 80)
    print("TASK 2 - VALIDATION")
    print("=" * 80)

    # 1. Check summary CSV and JSON
    csv_path = os.path.join(RESULTS_DIR, "reconstruction_summary.csv")
    json_path = os.path.join(RESULTS_DIR, "reconstruction_summary.json")

    assert os.path.exists(csv_path), f"Missing CSV summary: {csv_path}"
    assert os.path.exists(json_path), f"Missing JSON summary: {json_path}"

    df = pd.read_csv(csv_path)
    with open(json_path, "r") as f:
        data = json.load(f)

    print(f"[✓] Found summary files: {csv_path} and {json_path}")
    print(f"[✓] Records count: {len(df)}")

    # 2. Validate per-client files
    for cid in CLIENTS:
        recon_path = os.path.join(RESULTS_DIR, f"client{cid}_recon.png")
        orig_path = os.path.join(RESULTS_DIR, f"client{cid}_original.png")
        comp_path = os.path.join(RESULTS_DIR, f"client{cid}_comparison.png")
        prog_path = os.path.join(RESULTS_DIR, f"client{cid}_progression.png")
        loss_path = os.path.join(RESULTS_DIR, f"client{cid}_loss_curve.png")

        for p in [recon_path, orig_path, comp_path, prog_path, loss_path]:
            assert os.path.exists(p), f"Missing artifact: {p}"
            # Verify file is readable as image
            img = Image.open(p)
            assert img.size[0] > 0 and img.size[1] > 0

        # Check metrics from DataFrame
        row = df[df["client_id"] == cid].iloc[0]
        assert row["label_match"] == 1, f"Client {cid} label mismatch: {row['pred_label']} vs {row['true_label']}"
        assert row["mse"] <= 0.01, f"Client {cid} MSE exceeds threshold: {row['mse']}"
        assert row["psnr"] >= 20.0, f"Client {cid} PSNR below threshold: {row['psnr']}"
        assert row["success"] == True or row["success"] == "True", f"Client {cid} marked as failure"

        print(
            f"[✓] Client {cid}: Label={int(row['pred_label'])} (100% Match) | "
            f"MSE={row['mse']:.6e} | PSNR={row['psnr']:.2f} dB | SSIM={row['ssim']:.4f} | PASS"
        )

    # 3. Check overall summary image
    summary_grid = os.path.join(RESULTS_DIR, "all_clients_summary.png")
    assert os.path.exists(summary_grid), f"Missing summary grid: {summary_grid}"
    print(f"[✓] Found summary grid image: {summary_grid}")

    print("=" * 80)
    print("TASK 2 VALIDATION PASSED: ALL TESTS AND CHECKS SUCCEEDED")
    print("=" * 80)


if __name__ == "__main__":
    main()
