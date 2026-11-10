"""
Portable Dependency-Light Edge Inference Runner (NumPy-only compatible).
Designed for constrained ARM64 edge targets (Raspberry Pi OS Lite, Android Termux)
without requiring scikit-learn, SciPy, pandas, or joblib at inference time.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from dsp.feature_extractor import MCSAFeatureExtractor


class PortableDecisionTreeNode:
    """Evaluates an exported scikit-learn DecisionTreeClassifier JSON model in pure Python."""

    def __init__(self, json_path: Path):
        with open(json_path, "r") as f:
            self.tree = json.load(f)
        self.children_left = self.tree["children_left"]
        self.children_right = self.tree["children_right"]
        self.feature = self.tree["feature"]
        self.threshold = self.tree["threshold"]
        self.value = self.tree["value"]
        self.classes = self.tree["classes"]

    def predict(self, feat_vector: np.ndarray) -> tuple[int, float]:
        node = 0
        while self.children_left[node] != -1:
            feat_idx = self.feature[node]
            if float(feat_vector[feat_idx]) <= self.threshold[node]:
                node = self.children_left[node]
            else:
                node = self.children_right[node]

        counts = self.value[node][0]
        total = sum(counts) or 1.0
        best_idx = int(max(range(len(counts)), key=lambda i: counts[i]))
        return int(self.classes[best_idx]), float(counts[best_idx] / total)


def diagnose_portable(
    waveform: np.ndarray,
    tree_model: PortableDecisionTreeNode,
    extractor: MCSAFeatureExtractor,
    target_type: str = "binary",
) -> Dict[str, Any]:
    """Runs end-to-end NumPy DSP + portable JSON Decision Tree inference."""
    t0 = time.perf_counter()
    rms_val = float(np.sqrt(np.mean(waveform ** 2)))
    if rms_val < config.MIN_ACTIVE_CURRENT_A:
        return {
            "status": "Motor Off / Standby",
            "prediction_class": -1,
            "is_faulty": False,
            "confidence": 1.0,
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
            "fundamental_freq_hz": 0.0,
            "sideband_power_db": -55.0,
            "current_rms_a": rms_val,
        }
    feat_vec, feat_dict = extractor.extract_from_signal(waveform)
    pred_class, confidence = tree_model.predict(feat_vec)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    label_map = config.BINARY_CLASSES if target_type == "binary" else config.FAULT_CLASSES
    is_faulty = bool(pred_class == 1) if target_type == "binary" else bool(pred_class > 0)
    return {
        "status": label_map.get(pred_class, "Unknown"),
        "prediction_class": pred_class,
        "is_faulty": is_faulty,
        "confidence": confidence,
        "latency_ms": latency_ms,
        "fundamental_freq_hz": feat_dict["fundamental_freq_hz"],
        "sideband_power_db": feat_dict["sideband_mean_db_rel"],
        "current_rms_a": feat_dict["current_rms_a"],
    }


def main():
    parser = argparse.ArgumentParser(description="Portable ARM64 / Termux NumPy-Only Edge Demo")
    parser.add_argument("--target", choices=["binary", "multiclass"], default="binary")
    parser.add_argument("--stream", action="store_true", help="Replay validation windows sequentially like real-time monitoring")
    parser.add_argument("--interval", type=float, default=0.5, help="Display pause (sec) between streamed windows")
    args = parser.parse_args()

    suffix = f"_{args.target}" if args.target != "binary" else ""
    tree_path = config.MODELS_DIR / f"decision_tree_portable{suffix}.json"
    if not tree_path.exists():
        raise FileNotFoundError(f"Portable tree JSON missing at {tree_path}. Run models/train.py first.")

    sample_path = config.SAMPLES_DIR / "ieee_validation_samples.npz"
    if not sample_path.exists():
        raise FileNotFoundError(f"Portable validation bundle missing at {sample_path}.")

    tree = PortableDecisionTreeNode(tree_path)
    extractor = MCSAFeatureExtractor()
    pack = np.load(sample_path, allow_pickle=True)
    signals = pack["signals"]
    labels = pack["labels_bin"] if args.target == "binary" else pack["labels_multi"]
    metadata = list(pack["metadata"])

    print("=" * 78)
    print(f" PORTABLE EDGE NODE DIAGNOSTICS ({args.target.upper()} | NumPy-Only Tree JSON)")
    print("=" * 78)

    correct = 0
    latencies = []
    for sig, truth, meta in zip(signals, labels, metadata):
        res = diagnose_portable(sig, tree, extractor, target_type=args.target)
        is_ok = int(res["prediction_class"]) == int(truth)
        correct += int(is_ok)
        latencies.append(res["latency_ms"])
        mark = "OK" if is_ok else "MISS"
        print(
            f"[{mark:4s}] Torque {meta['load_torque_nm']:3.1f} Nm | True Seve={meta['fault_severity']} BRB | "
            f"Pred={res['status']:26s} | SB={res['sideband_power_db']:6.2f} dB | {res['latency_ms']:5.2f} ms"
        )
        if args.stream:
            time.sleep(args.interval)

    acc = 100.0 * correct / len(signals)
    print("-" * 78)
    print(f"Summary across {len(signals)} unseen IEEE samples: Accuracy={acc:.1f}% | Mean Latency={np.mean(latencies):.2f} ms")


if __name__ == "__main__":
    main()
