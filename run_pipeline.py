"""
Master Pipeline Runner for Edge-Based Fault Diagnosis of Induction Motors.
Executes the end-to-end workflow:
1. Data Acquisition & Preprocessing (Treml et al. 2020 IEEE DataPort / MCSA Generator)
2. SciPy Welch PSD & Feature Extraction (MATLAB pwelch equivalent)
3. Machine Learning Model Training (SVM & Decision Tree)
4. Local Edge Deployment & Inference Verification
5. Empirical Benchmarking (Edge vs AWS EC2 Cloud)
6. Publication Figures & Interactive Dashboard Generation
"""

import sys
import argparse
from pathlib import Path
import numpy as np

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from data.dataset_loader import StatorCurrentDataset
from models.train import ModelTrainer
from models.evaluate import run_robustness_audit
from edge.edge_runtime import EdgeFaultClassifier
from edge.benchmark import EdgeCloudBenchmark
from visualization.plot_signals import generate_all_plots
from visualization.dashboard import generate_html_dashboard


def _export_portable_samples(dataset: StatorCurrentDataset) -> None:
    """Saves a lightweight (<200 KB) unseen-recording sample bundle for portable ARM64 / CI execution."""
    out_path = config.SAMPLES_DIR / "ieee_validation_samples.npz"
    signals_all, labels_m_all, labels_bin_all, meta_all = dataset.load_processed_dataset()
    selected_idx = []
    for torque in config.LOAD_TORQUES_NM:
        h_cand = [
            i for i, m in enumerate(meta_all)
            if m["load_torque_nm"] == torque and m["fault_severity"] == 0 and m.get("repetition") in config.HOLDOUT_REPETITIONS
        ]
        f_cand = [
            i for i, m in enumerate(meta_all)
            if m["load_torque_nm"] == torque and m["fault_severity"] == 3 and m.get("repetition") in config.HOLDOUT_REPETITIONS
        ]
        if h_cand:
            selected_idx.append(h_cand[0])
        if f_cand:
            selected_idx.append(f_cand[0])
    if selected_idx:
        np.savez_compressed(
            out_path,
            signals=signals_all[selected_idx],
            labels_multi=labels_m_all[selected_idx],
            labels_bin=labels_bin_all[selected_idx],
            metadata=np.array([meta_all[i] for i in selected_idx], dtype=object),
        )


def main():
    parser = argparse.ArgumentParser(
        description="Edge-Based Induction Motor Fault Diagnosis Pipeline (Open-Source Linux / ARM64 Edge)"
    )
    parser.add_argument(
        "--samples", type=int, default=30,
        help="Number of signal samples per condition (5 loads x 5 severities = 25 conditions)"
    )
    parser.add_argument(
        "--force-regen", action="store_true",
        help="Force regeneration of signals and feature extraction"
    )
    parser.add_argument(
        "--bench-runs", type=int, default=150,
        help="Number of iterations for Edge vs Cloud benchmark"
    )
    parser.add_argument(
        "--skip-bench", action="store_true",
        help="Skip benchmark phase"
    )
    parser.add_argument(
        "--skip-plots", action="store_true",
        help="Skip figure and dashboard generation"
    )

    args = parser.parse_args()

    print("=======================================================================")
    print("   EDGE-BASED FAULT DIAGNOSIS OF INDUCTION MOTORS                     ")
    print("   Lightweight Edge-AI Architecture (Local Linux / ARM64)             ")
    print("   Author: Sai Samanyu K (231CS152) - NITK                            ")
    print("   Ref: Walani & Doorsamy (2025), Treml et al. (2020)                 ")
    print("=======================================================================")

    # STEP 1 & 2: Data Acquisition & Feature Extraction
    print("\n>>> [PHASE 1 & 2] Data Acquisition & Hardened Welch PSD Extraction...")
    trainer = ModelTrainer()
    X, y_bin, y_multi, df_features = trainer.prepare_data(
        samples_per_condition=args.samples,
        force_regenerate=args.force_regen
    )
    print(f"Total dataset size: {len(X)} samples, {X.shape[1]} scale/grid-normalized MCSA features.")

    # STEP 3: Train Edge-Viable Machine Learning Models (Zero Within-Recording Leakage)
    print("\n>>> [PHASE 3] Machine Learning Model Training (Recording-Grouped)...")
    results_bin = trainer.train_models(X, y_bin, target_type="binary", df_meta=df_features)
    results_multi = trainer.train_models(X, y_multi, target_type="multiclass", df_meta=df_features)

    # STEP 4: Robustness Stress Audit & Edge Inference Verification
    print("\n>>> [PHASE 4] Verifying Robustness & Local Edge Inference Engine...")
    dataset = StatorCurrentDataset()
    dataset_file = config.PROCESSED_DATA_DIR / "stator_current_dataset.npz"
    if dataset_file.exists():
        _export_portable_samples(dataset)
        run_robustness_audit()
        signals_all, _, labels_bin_all, meta_all = dataset.load_processed_dataset()
        h_indices = [i for i, m in enumerate(meta_all) if m["fault_severity"] == 0 and m.get("repetition") in config.HOLDOUT_REPETITIONS]
        f_indices = [i for i, m in enumerate(meta_all) if m["fault_severity"] >= 3 and m.get("repetition") in config.HOLDOUT_REPETITIONS]
        test_healthy = signals_all[h_indices[0] if h_indices else int(np.where(labels_bin_all == 0)[0][0])]
        test_faulty = signals_all[f_indices[0] if f_indices else int(np.where(labels_bin_all == 1)[0][0])]
    else:
        test_healthy, _ = dataset.generate_single_signal(fault_severity=0, load_torque_nm=2.0)
        test_faulty, _ = dataset.generate_single_signal(fault_severity=3, load_torque_nm=2.0)

    edge_dt = EdgeFaultClassifier(model_type="decision_tree", target_type="binary")
    pred_h = edge_dt.diagnose_waveform(test_healthy)
    pred_f = edge_dt.diagnose_waveform(test_faulty)

    print(f"  Unseen Healthy: Predicted '{pred_h['status']}' in {pred_h['latency']['total_edge_ms']:.2f} ms")
    print(f"  Unseen Faulty:  Predicted '{pred_f['status']}' in {pred_f['latency']['total_edge_ms']:.2f} ms")

    # STEP 5: Benchmarking (Measured Edge vs Simulated Cloud)
    if not args.skip_bench:
        print("\n>>> [PHASE 5] Benchmarking (Measured Local Edge vs Simulated Cloud)...")
        benchmark = EdgeCloudBenchmark(num_benchmark_runs=args.bench_runs)
        benchmark.run_full_benchmark()

    # STEP 6: Figures & Dashboard
    if not args.skip_plots:
        print("\n>>> [PHASE 6] Generating Visualizations and HTML Dashboard...")
        generate_all_plots()
        dashboard_path = generate_html_dashboard()
        print(f"Dashboard available at: file://{dashboard_path}")

    print("\n=======================================================================")
    print("   PIPELINE COMPLETED SUCCESSFULLY!                                   ")
    print("=======================================================================")
    print(f"Serialized Models:    {config.MODELS_DIR}")
    print(f"Portable Edge Runner: {config.BASE_DIR / 'edge' / 'portable_edge.py'}")
    print(f"Figures & Plots:      {config.PLOTS_DIR}")
    print(f"Interactive Report:   file://{config.REPORTS_DIR / 'index.html'}")
    print("=======================================================================")


if __name__ == "__main__":
    main()
