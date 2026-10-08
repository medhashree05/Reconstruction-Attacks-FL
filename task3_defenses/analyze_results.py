import pandas as pd


RESULTS_FILE = "task3_defenses/results/reconstruction_defense_results.csv"


def main():

    df = pd.read_csv(RESULTS_FILE)

    print("=" * 80)
    print("TASK 3 - RECONSTRUCTION DEFENSE RESULTS ANALYSIS")
    print("=" * 80)

    # --------------------------------------------------------
    # Overall results
    # --------------------------------------------------------

    print("\nOVERALL RESULTS")
    print("-" * 80)

    print(f"Total experiments : {len(df)}")

    print(
        f"Average MSE       : "
        f"{df['mse'].mean():.8f}"
    )

    print(
        f"Average PSNR      : "
        f"{df['psnr'].mean():.4f}"
    )

    print(
        f"Average SSIM      : "
        f"{df['ssim'].mean():.6f}"
    )

    print(
        f"Attack success    : "
        f"{df['success'].mean() * 100:.2f}%"
    )

    print(
        f"Label recovery    : "
        f"{df['label_match'].mean() * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Gradient clipping
    # --------------------------------------------------------

    clipping = df[df["defense"] == "clipping"].copy()

    print("\n")
    print("=" * 80)
    print("GRADIENT CLIPPING RESULTS")
    print("=" * 80)

    clipping_summary = (
        clipping
        .groupby("parameter")
        .agg(
            experiments=("mse", "count"),
            avg_mse=("mse", "mean"),
            avg_psnr=("psnr", "mean"),
            avg_ssim=("ssim", "mean"),
            attack_success_rate=("success", "mean"),
            label_recovery_rate=("label_match", "mean")
        )
        .reset_index()
    )

    clipping_summary["attack_success_rate"] *= 100
    clipping_summary["label_recovery_rate"] *= 100

    print(
        clipping_summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Differential privacy
    # --------------------------------------------------------

    dp = df[df["defense"] == "dp"].copy()

    print("\n")
    print("=" * 80)
    print("DIFFERENTIAL PRIVACY RESULTS")
    print("=" * 80)

    dp_summary = (
        dp
        .groupby("parameter")
        .agg(
            experiments=("mse", "count"),
            avg_mse=("mse", "mean"),
            avg_psnr=("psnr", "mean"),
            avg_ssim=("ssim", "mean"),
            attack_success_rate=("success", "mean"),
            label_recovery_rate=("label_match", "mean")
        )
        .reset_index()
    )

    dp_summary["attack_success_rate"] *= 100
    dp_summary["label_recovery_rate"] *= 100

    print(
        dp_summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Client-wise results
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("CLIENT-WISE RESULTS")
    print("=" * 80)

    client_summary = (
        df
        .groupby(["defense", "parameter"])
        .agg(
            avg_mse=("mse", "mean"),
            avg_psnr=("psnr", "mean"),
            avg_ssim=("ssim", "mean"),
            attack_success_rate=("success", "mean"),
            label_recovery_rate=("label_match", "mean")
        )
        .reset_index()
    )

    client_summary["attack_success_rate"] *= 100
    client_summary["label_recovery_rate"] *= 100

    print(
        client_summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Best defenses
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("DEFENSE COMPARISON")
    print("=" * 80)

    print("\nHighest average MSE:")
    best_mse = df.loc[df["mse"].idxmax()]
    print(
        f"{best_mse['defense']} "
        f"{best_mse['parameter']} "
        f"-> MSE = {best_mse['mse']:.8f}"
    )

    print("\nLowest average PSNR:")
    best_psnr = df.loc[df["psnr"].idxmin()]
    print(
        f"{best_psnr['defense']} "
        f"{best_psnr['parameter']} "
        f"-> PSNR = {best_psnr['psnr']:.4f}"
    )

    print("\nLowest average SSIM:")
    best_ssim = df.loc[df["ssim"].idxmin()]
    print(
        f"{best_ssim['defense']} "
        f"{best_ssim['parameter']} "
        f"-> SSIM = {best_ssim['ssim']:.6f}"
    )

    # --------------------------------------------------------
    # Save summaries
    # --------------------------------------------------------

    clipping_summary.to_csv(
        "task3_defenses/results/clipping_reconstruction_summary.csv",
        index=False
    )

    dp_summary.to_csv(
        "task3_defenses/results/dp_reconstruction_summary.csv",
        index=False
    )

    client_summary.to_csv(
        "task3_defenses/results/defense_comparison_summary.csv",
        index=False
    )

    print("\n")
    print("=" * 80)
    print("SUMMARY FILES SAVED")
    print("=" * 80)

    print(
        "task3_defenses/results/"
        "clipping_reconstruction_summary.csv"
    )

    print(
        "task3_defenses/results/"
        "dp_reconstruction_summary.csv"
    )

    print(
        "task3_defenses/results/"
        "defense_comparison_summary.csv"
    )


if __name__ == "__main__":
    main()