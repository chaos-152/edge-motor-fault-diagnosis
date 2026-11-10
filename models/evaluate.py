"""Model evaluation, metrics reporting, and real-world robustness stress suite."""
import json
import sys
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from data.dataset_loader import StatorCurrentDataset
from dsp.feature_extractor import MCSAFeatureExtractor


def evaluate_model_pipeline(model_path: Path, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """Evaluates serialized model on test data and returns detailed metrics."""
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy": float(report_dict["accuracy"]),
        "weighted_avg": report_dict["weighted avg"],
        "confusion_matrix": cm.tolist(),
        "classes": np.unique(y_test).tolist()
    }

    if y_prob is not None and len(np.unique(y_test)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob[:, 1]))

    return metrics


def run_robustness_audit(output_file: Path = config.REPORTS_DIR / "robustness_metrics.json") -> Dict[str, Any]:
    """
    Evaluates trained binary Decision Tree and SVM against realistic domain shifts
    on strictly unseen experimental recordings (repetitions 8 & 9):
      1. Clean holdout baseline
      2. Calibrated sensor scaling (x2.5 amplitude + matched nameplate calibration)
      3. Uncalibrated sensor gain drift (x1.5 and x0.5 amplitude without recalibration)
      4. Additive industrial Gaussian noise (sigma = 0.05 A and 0.08 A)
      5. Cross-grid 50 Hz synthetic sanity test (on 60 Hz-trained models)
    """
    dt_model = joblib.load(config.MODELS_DIR / "decision_tree_model.joblib")
    svm_model = joblib.load(config.MODELS_DIR / "svm_model.joblib")

    dataset = StatorCurrentDataset()
    dataset_file = config.PROCESSED_DATA_DIR / "stator_current_dataset.npz"
    if not dataset_file.exists():
        raise FileNotFoundError(f"Processed dataset missing at {dataset_file}; run training first.")

    signals, _, y_bin, metadata = dataset.load_processed_dataset()
    reps = np.array([int(m.get("repetition", idx)) for idx, m in enumerate(metadata)])
    holdout_mask = np.isin(reps, config.HOLDOUT_REPETITIONS)
    if not np.any(holdout_mask):
        holdout_mask = np.ones(len(signals), dtype=bool)

    sig_te = signals[holdout_mask]
    y_te = y_bin[holdout_mask]

    extractor_std = MCSAFeatureExtractor()
    extractor_cal25 = MCSAFeatureExtractor(
        i_noload_a=config.MOTOR_NOLOAD_CURRENT_A * 2.5,
        i_rated_a=config.MOTOR_RATED_CURRENT_A * 2.5,
    )

    rng = np.random.default_rng(config.RANDOM_STATE)
    scenarios = {
        "clean_unseen_recordings": extractor_std.extract_batch(sig_te),
        "calibrated_gain_x2.5": extractor_cal25.extract_batch(sig_te * 2.5),
        "uncalibrated_gain_x1.5": extractor_std.extract_batch(sig_te * 1.5),
        "uncalibrated_gain_x0.5": extractor_std.extract_batch(sig_te * 0.5),
        "noise_sigma_0.05A": extractor_std.extract_batch(
            sig_te + rng.normal(0.0, 0.05, sig_te.shape).astype(np.float32)
        ),
        "noise_sigma_0.08A": extractor_std.extract_batch(
            sig_te + rng.normal(0.0, 0.08, sig_te.shape).astype(np.float32)
        ),
    }

    report: Dict[str, Any] = {
        "n_unseen_test_samples": int(len(y_te)),
        "holdout_repetitions": list(config.HOLDOUT_REPETITIONS),
        "scenarios": {},
    }

    print("\n=== REAL-WORLD ROBUSTNESS & DOMAIN-SHIFT AUDIT ===")
    for name, X_scen in scenarios.items():
        acc_dt = float(accuracy_score(y_te, dt_model.predict(X_scen)))
        acc_svm = float(accuracy_score(y_te, svm_model.predict(X_scen)))
        report["scenarios"][name] = {"DecisionTree": acc_dt, "SVM": acc_svm}
        print(f"  {name:26s} | DT: {acc_dt*100:6.2f}% | SVM: {acc_svm*100:6.2f}%")

    # Cross-grid 50 Hz synthetic check (motor running on 50 Hz power grid with slip s=0.04)
    t_ax = np.linspace(0.0, config.WINDOW_DURATION_SEC, int(config.WINDOW_DURATION_SEC * config.TARGET_SAMPLING_RATE_HZ), endpoint=False)
    sig_50_h = (2.2 * np.sin(2 * np.pi * 50.0 * t_ax) + 0.015 * rng.normal(size=len(t_ax))).astype(np.float32)
    sig_50_f = (
        2.2 * np.sin(2 * np.pi * 50.0 * t_ax)
        + 0.16 * np.sin(2 * np.pi * 50.0 * (1 - 2 * 0.04) * t_ax)
        + 0.14 * np.sin(2 * np.pi * 50.0 * (1 + 2 * 0.04) * t_ax)
        + 0.015 * rng.normal(size=len(t_ax))
    ).astype(np.float32)
    X_50 = extractor_std.extract_batch(np.stack([sig_50_h, sig_50_f]))
    y_50 = np.array([0, 1])
    acc_50_dt = float(accuracy_score(y_50, dt_model.predict(X_50)))
    acc_50_svm = float(accuracy_score(y_50, svm_model.predict(X_50)))
    report["scenarios"]["synthetic_50hz_cross_grid"] = {
        "DecisionTree": acc_50_dt,
        "SVM": acc_50_svm,
        "n_samples": 2,
    }
    print(f"  {'synthetic_50hz_cross_grid':26s} | DT: {acc_50_dt*100:6.2f}% | SVM: {acc_50_svm*100:6.2f}% (n=2)")

    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Robustness audit saved to: {output_file}")
    return report


if __name__ == "__main__":
    run_robustness_audit()
