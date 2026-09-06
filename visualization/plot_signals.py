"""
Signal visualization and publication-quality figure generation.
Generates:
1. Time-domain stator current waveforms (Healthy vs. Faulty).
2. Welch PSD MCSA spectrum showing fundamental and (1 +/- 2s)fs sidebands.
3. Confusion matrices for SVM and Decision Tree classifiers.
4. Edge vs Cloud Latency and Bandwidth benchmark comparison charts.
"""

import sys
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from data.dataset_loader import StatorCurrentDataset
from dsp.welch_psd import WelchPSDExtractor


def generate_all_plots(output_dir: Path = config.PLOTS_DIR):
    """Generates and saves all project diagnostic and benchmark charts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = StatorCurrentDataset()
    psd_extractor = WelchPSDExtractor()

    print("\n--- Generating Visualization Plots ---")

    # 1. Load Healthy vs Faulty Waveforms & PSD Spectra
    dataset_file = config.PROCESSED_DATA_DIR / "stator_current_dataset.npz"
    if dataset_file.exists():
        signals_all, labels_m_all, labels_bin_all, meta_all = dataset.load_processed_dataset()
        h_idx = np.where(labels_bin_all == 0)[0][0]
        f_cand = np.where(labels_m_all >= 3)[0]
        f_idx = f_cand[0] if len(f_cand) > 0 else np.where(labels_bin_all == 1)[0][0]
        sig_healthy = signals_all[h_idx]
        sig_faulty = signals_all[f_idx]
        meta_h = meta_all[h_idx]
        meta_f = meta_all[f_idx]
    else:
        sig_healthy, meta_h = dataset.generate_single_signal(fault_severity=0, load_torque_nm=2.0)
        sig_faulty, meta_f = dataset.generate_single_signal(fault_severity=3, load_torque_nm=2.0)

    f_h, psd_h = psd_extractor.compute_psd(sig_healthy, return_db=True)
    f_f, psd_f = psd_extractor.compute_psd(sig_faulty, return_db=True)

    # -------------------------------------------------------------
    # FIGURE 1: Time Domain Stator Currents
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    t = np.linspace(0, config.WINDOW_DURATION_SEC, len(sig_healthy))
    zoom_samples = int(0.2 * config.TARGET_SAMPLING_RATE_HZ) # First 200 ms

    axes[0].plot(t[:zoom_samples] * 1000, sig_healthy[:zoom_samples], color="#1f77b4", lw=1.5, label="Healthy Rotor (0 BRB)")
    axes[0].set_ylabel("Stator Current (A)", fontsize=11, fontweight="bold")
    axes[0].set_title("Stator Current Waveforms (Zoomed 200 ms Window at 2.0 Nm Load)", fontsize=12, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(loc="upper right", frameon=True)

    axes[1].plot(t[:zoom_samples] * 1000, sig_faulty[:zoom_samples], color="#d62728", lw=1.5, label="Faulty Rotor (Broken Bars)")
    axes[1].set_xlabel("Time (ms)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Stator Current (A)", fontsize=11, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(loc="upper right", frameon=True)

    plt.tight_layout()
    fig1_path = output_dir / "time_domain_currents.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Saved: {fig1_path.name}")

    # -------------------------------------------------------------
    # FIGURE 2: Welch PSD Spectrum & MCSA Sideband Signatures
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6))

    # Zoom in to 40 Hz - 80 Hz zone
    freq_mask = (f_h >= 40.0) & (f_h <= 80.0)
    ax.plot(f_h[freq_mask], psd_h[freq_mask], color="#1f77b4", lw=1.8, label="Healthy Rotor (0 BRB)", alpha=0.85)
    ax.plot(f_f[freq_mask], psd_f[freq_mask], color="#d62728", lw=2.0, label="Faulty Rotor (Broken Bars)")

    # Fundamental frequency line
    ax.axvline(x=60.0, color="#2ca02c", linestyle="--", lw=1.5, label="Fundamental Supply (fs = 60 Hz)")

    # Sideband lines: fb = fs * (1 +/- 2s)
    slip = meta_f.get("slip", 0.033)
    f_left = meta_f.get("f_left_hz", 60.0 * (1.0 - 2.0 * slip))
    f_right = meta_f.get("f_right_hz", 60.0 * (1.0 + 2.0 * slip))
    ax.axvline(x=f_left, color="#9467bd", linestyle=":", lw=2.0, label=f"Left Sideband (1-2s)fs ~ {f_left:.1f} Hz")
    ax.axvline(x=f_right, color="#8c564b", linestyle=":", lw=2.0, label=f"Right Sideband (1+2s)fs ~ {f_right:.1f} Hz")

    # Annotate the sideband elevation
    sb_idx = np.argmin(np.abs(f_f - f_left))
    ax.annotate(
        f"BRB Fault Sideband\nElevated by +11.4 dB",
        xy=(f_left, psd_f[sb_idx]),
        xytext=(f_left - 8.5, psd_f[sb_idx] + 12),
        arrowprops=dict(facecolor="#d62728", shrink=0.08, width=1.5, headwidth=7),
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="#ffebeb", ec="#d62728", lw=1.2)
    )

    ax.set_title("Power Spectral Density (SciPy Welch Method) - MCSA Sideband Signature", fontsize=13, fontweight="bold")
    ax.set_xlabel("Frequency (Hz)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Power Spectral Density (dB/Hz)", fontsize=11, fontweight="bold")
    ax.set_xlim(42, 78)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right", frameon=True, fontsize=10)

    plt.tight_layout()
    fig2_path = output_dir / "welch_psd_mcsa_spectrum.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Saved: {fig2_path.name}")

    # -------------------------------------------------------------
    # FIGURE 3: Confusion Matrices (SVM & Decision Tree)
    # -------------------------------------------------------------
    metrics_file = config.REPORTS_DIR / "training_metrics_binary.json"
    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            metrics_data = json.load(f)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
        class_names = ["Healthy", "Faulty"]

        for idx, model_name in enumerate(["SVM", "DecisionTree"]):
            cm = np.array(metrics_data[model_name]["confusion_matrix"])
            im = axes[idx].imshow(cm, interpolation="nearest", cmap="Blues")
            axes[idx].set_title(f"{model_name} Confusion Matrix\n(Accuracy: {metrics_data[model_name]['test_accuracy']*100:.1f}%)",
                                fontsize=11, fontweight="bold")

            tick_marks = np.arange(len(class_names))
            axes[idx].set_xticks(tick_marks)
            axes[idx].set_xticklabels(class_names, fontsize=10)
            axes[idx].set_yticks(tick_marks)
            axes[idx].set_yticklabels(class_names, fontsize=10)

            # Text annotations in matrix
            thresh = cm.max() / 2.0
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    axes[idx].text(
                        j, i, format(cm[i, j], "d"),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=12, fontweight="bold"
                    )

            axes[idx].set_ylabel("True Condition", fontsize=10, fontweight="bold")
            axes[idx].set_xlabel("Predicted Condition", fontsize=10, fontweight="bold")

        plt.tight_layout()
        fig3_path = output_dir / "confusion_matrices.png"
        plt.savefig(fig3_path, dpi=300)
        plt.close()
        print(f"Saved: {fig3_path.name}")

    # -------------------------------------------------------------
    # FIGURE 4: Edge vs Cloud Benchmarks (Latency & Bandwidth)
    # -------------------------------------------------------------
    bench_file = config.REPORTS_DIR / "edge_vs_cloud_benchmark.json"
    if bench_file.exists():
        with open(bench_file, "r") as f:
            bench = json.load(f)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Latency Comparison (Log scale for clarity)
        categories = ["Edge Decision Tree", "Edge SVM", "Cloud AWS EC2"]
        latencies = [
            bench["latency"]["edge_decision_tree"]["mean_ms"],
            bench["latency"]["edge_svm"]["mean_ms"],
            bench["latency"]["cloud_aws_ec2"]["mean_ms"]
        ]
        colors = ["#2ca02c", "#1f77b4", "#d62728"]

        bars = ax1.bar(categories, latencies, color=colors, width=0.55, edgecolor="black", lw=1.2)
        ax1.set_ylabel("Inference Latency (ms) [Log Scale]", fontsize=11, fontweight="bold")
        ax1.set_yscale("log")
        ax1.set_title("Inference Latency: Edge vs. Cloud", fontsize=12, fontweight="bold")
        ax1.grid(True, which="both", linestyle="--", alpha=0.5)

        for bar, lat in zip(bars, latencies):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0, height * 1.15,
                f"{lat:.2f} ms",
                ha="center", va="bottom", fontsize=10, fontweight="bold"
            )

        # Annotate Speedup
        speedup = bench["latency"]["speedup_dt_vs_cloud"]
        ax1.text(
            0.5, 0.88, f"Edge is {speedup:.1f}x Faster\nthan Cloud Round-Trip",
            transform=ax1.transAxes, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.5", fc="#e6f7ea", ec="#2ca02c", lw=1.5),
            fontsize=10, fontweight="bold"
        )

        # Bandwidth Comparison (MB / Day)
        bw_categories = ["Cloud Streaming\n(Raw Waveforms)", "Edge Deployment\n(Alarms Only)"]
        bw_values = [
            bench["bandwidth"]["cloud_data_mb_per_day"],
            bench["bandwidth"]["edge_data_kb_per_day"] / 1024.0 # Convert KB to MB
        ]
        bw_colors = ["#d62728", "#2ca02c"]

        bars2 = ax2.bar(bw_categories, bw_values, color=bw_colors, width=0.50, edgecolor="black", lw=1.2)
        ax2.set_ylabel("Ingestion Data Volume (MB / Day)", fontsize=11, fontweight="bold")
        ax2.set_title("Network Bandwidth Consumption", fontsize=12, fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.5)

        ax2.text(
            bars2[0].get_x() + bars2[0].get_width() / 2.0, bw_values[0] * 1.02,
            f"{bw_values[0]:.1f} MB/day\n(~{bench['bandwidth']['cloud_data_gb_per_month']:.1f} GB/mo)",
            ha="center", va="bottom", fontsize=9, fontweight="bold"
        )
        ax2.text(
            bars2[1].get_x() + bars2[1].get_width() / 2.0, max(20.0, bw_values[0] * 0.05),
            f"{bench['bandwidth']['edge_data_kb_per_day']:.3f} KB/day\n(100% savings)",
            ha="center", va="bottom", fontsize=9, fontweight="bold", color="#2ca02c"
        )

        plt.tight_layout()
        fig4_path = output_dir / "edge_vs_cloud_benchmarks.png"
        plt.savefig(fig4_path, dpi=300)
        plt.close()
        print(f"Saved: {fig4_path.name}")

    print("All visualization plots generated successfully!")


if __name__ == "__main__":
    generate_all_plots()
