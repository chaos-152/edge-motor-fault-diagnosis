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
from edge.edge_runtime import EdgeFaultClassifier
from edge.benchmark import EdgeCloudBenchmark
from visualization.plot_signals import generate_all_plots
from visualization.dashboard import generate_html_dashboard


def main():
    parser = argparse.ArgumentParser(
        description="Edge-Based Induction Motor Fault Diagnosis Pipeline (Raspberry Pi & Open-Source Python)"
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
    print("   Lightweight Machine Learning on Raspberry Pi                       ")
    print("   Author: Sai Samanyu K (231CS152) - NITK                            ")
    print("   Ref: Walani & Doorsamy (2025), Treml et al. (2020)                 ")
    print("=======================================================================")

    # STEP 1 & 2: Data Acquisition & Feature Extraction
    print("\n>>> [PHASE 1 & 2] Data Acquisition & SciPy Welch PSD Extraction...")
    trainer = ModelTrainer()
    X, y_bin, y_multi, df_features = trainer.prepare_data(
        samples_per_condition=args.samples,
        force_regenerate=args.force_regen
    )
    print(f"Total dataset size: {len(X)} samples, {X.shape[1]} MCSA features per sample.")

    # STEP 3: Train Edge-Viable Machine Learning Models
    print("\n>>> [PHASE 3] Machine Learning Model Training (SVM & Decision Tree)...")
    results_bin = trainer.train_models(X, y_bin, target_type="binary")
    results_multi = trainer.train_models(X, y_multi, target_type="multiclass")

    # STEP 4: Edge Deployment & Single-Shot Verification
    print("\n>>> [PHASE 4] Verifying Local Edge Inference Engine...")
    edge_dt = EdgeFaultClassifier(model_type="decision_tree", target_type="binary")
    edge_svm = EdgeFaultClassifier(model_type="svm", target_type="binary")

    dataset = StatorCurrentDataset()
    dataset_file = config.PROCESSED_DATA_DIR / "stator_current_dataset.npz"
    if dataset_file.exists():
        signals_all, _, labels_bin_all, _ = dataset.load_processed_dataset()
        h_indices = np.where(labels_bin_all == 0)[0]
        f_indices = np.where(labels_bin_all == 1)[0]
        test_healthy = signals_all[h_indices[0]]
        test_faulty = signals_all[f_indices[0]]
    else:
        test_healthy, _ = dataset.generate_single_signal(fault_severity=0, load_torque_nm=2.0)
        test_faulty, _ = dataset.generate_single_signal(fault_severity=2, load_torque_nm=2.0)

    pred_h = edge_dt.diagnose_waveform(test_healthy)
    pred_f = edge_dt.diagnose_waveform(test_faulty)

    print(f"  Test Healthy: Predicted '{pred_h['status']}' in {pred_h['latency']['total_edge_ms']:.2f} ms")
    print(f"  Test Faulty:  Predicted '{pred_f['status']}' in {pred_f['latency']['total_edge_ms']:.2f} ms")

    # STEP 5: Benchmarking (Edge vs Cloud)
    if not args.skip_bench:
        print("\n>>> [PHASE 5] Empirical Benchmarking (Edge vs AWS EC2 Cloud)...")
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
    print(f"Serialized Models:   {config.MODELS_DIR}")
    print(f"Figures & Plots:     {config.PLOTS_DIR}")
    print(f"Interactive Report:  file://{config.REPORTS_DIR / 'index.html'}")
    print(f"Pi Deployment Guide: {config.BASE_DIR / 'edge' / 'deploy_pi.sh'}")
    print("=======================================================================")


if __name__ == "__main__":
    main()
