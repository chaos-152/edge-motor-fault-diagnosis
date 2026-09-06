"""
Edge Runtime Engine for Raspberry Pi Model 4 and Local Edge Inference.
Runs completely offline without cloud dependencies or network overhead.
Ref: "Edge vs. Cloud: Empirical Insights into Data-Driven Condition Monitoring" (Walani & Doorsamy, 2025)
"""

import sys
import time
from pathlib import Path
from typing import Dict, Optional, Union

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from dsp.welch_psd import WelchPSDExtractor
from dsp.feature_extractor import MCSAFeatureExtractor


class EdgeFaultClassifier:
    """
    Lightweight, high-performance edge diagnostic engine for Raspberry Pi.
    """

    def __init__(
        self,
        model_type: str = "decision_tree",
        target_type: str = "binary",
        models_dir: Path = config.MODELS_DIR
    ):
        self.model_type = model_type.lower()
        self.target_type = target_type.lower()
        self.models_dir = Path(models_dir)

        # Initialize DSP pipeline
        self.psd_extractor = WelchPSDExtractor()
        self.feature_extractor = MCSAFeatureExtractor(psd_extractor=self.psd_extractor)

        # Load serialized model
        self.model = self._load_model()

    def _load_model(self):
        suffix = f"_{self.target_type}" if self.target_type != "binary" else ""
        if "tree" in self.model_type or "dt" in self.model_type:
            filename = f"decision_tree_model{suffix}.joblib"
        elif "svm" in self.model_type:
            filename = f"svm_model{suffix}.joblib"
        else:
            raise ValueError(f"Unsupported model type '{self.model_type}'. Choose 'decision_tree' or 'svm'.")

        model_path = self.models_dir / filename
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {model_path}. Run 'python models/train.py' first."
            )

        return joblib.load(model_path)

    def diagnose_waveform(
        self,
        stator_current: np.ndarray,
        estimated_slip: Optional[float] = None
    ) -> Dict[str, Union[str, int, float, Dict]]:
        """
        Executes complete end-to-end edge inference on a stator current window:
        1. SciPy Welch PSD calculation.
        2. MCSA physical sideband & statistical feature extraction.
        3. Scikit-learn Classifier inference.
        4. Detailed latency profiling.
        """
        t0 = time.perf_counter()

        # Step 1: PSD Computation
        t_psd_start = time.perf_counter()
        frequencies, psd_db = self.psd_extractor.compute_psd(stator_current, return_db=True)
        _, psd_lin = self.psd_extractor.compute_psd(stator_current, return_db=False)
        t_psd_ms = (time.perf_counter() - t_psd_start) * 1000.0

        # Step 2: Feature Extraction
        t_feat_start = time.perf_counter()
        feat_vector, feat_dict = self.feature_extractor.extract_from_psd(
            frequencies, psd_db, psd_lin, signal_array=stator_current, estimated_slip=estimated_slip
        )
        t_feat_ms = (time.perf_counter() - t_feat_start) * 1000.0

        # Step 3: Classifier Inference
        t_infer_start = time.perf_counter()
        feat_matrix = feat_vector.reshape(1, -1)
        pred_label = int(self.model.predict(feat_matrix)[0])

        confidence = 1.0
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(feat_matrix)[0]
            confidence = float(np.max(probs))
        t_infer_ms = (time.perf_counter() - t_infer_start) * 1000.0

        total_edge_latency_ms = (time.perf_counter() - t0) * 1000.0

        # Label resolution
        if self.target_type == "binary":
            status_text = config.BINARY_CLASSES.get(pred_label, "Unknown")
            is_faulty = bool(pred_label == 1)
        else:
            status_text = config.FAULT_CLASSES.get(pred_label, "Unknown")
            is_faulty = bool(pred_label > 0)

        return {
            "status": status_text,
            "prediction_class": pred_label,
            "is_faulty": is_faulty,
            "confidence": confidence,
            "latency": {
                "psd_ms": t_psd_ms,
                "feature_extraction_ms": t_feat_ms,
                "classifier_ms": t_infer_ms,
                "total_edge_ms": total_edge_latency_ms
            },
            "sideband_power_db": feat_dict["sideband_mean_db_rel"],
            "fault_energy_ratio": feat_dict["fault_band_energy_ratio"],
            "current_rms_a": feat_dict["current_rms_a"]
        }


if __name__ == "__main__":
    from data.dataset_loader import StatorCurrentDataset
    dataset = StatorCurrentDataset()

    print("Initializing Edge Diagnostic Engine (Decision Tree)...")
    classifier = EdgeFaultClassifier(model_type="decision_tree", target_type="binary")

    # Test on Healthy Signal
    sig_h, _ = dataset.generate_single_signal(fault_severity=0, load_torque_nm=2.0)
    res_h = classifier.diagnose_waveform(sig_h)
    print("\n--- Diagnostic Result: Healthy Motor ---")
    print(f"Status:             {res_h['status']}")
    print(f"Sideband Relative:  {res_h['sideband_power_db']:.2f} dB")
    print(f"Total Edge Latency: {res_h['latency']['total_edge_ms']:.3f} ms")

    # Test on Faulty Signal (3 Broken Bars)
    sig_f, _ = dataset.generate_single_signal(fault_severity=3, load_torque_nm=2.0)
    res_f = classifier.diagnose_waveform(sig_f)
    print("\n--- Diagnostic Result: Faulty Motor (3 BRB) ---")
    print(f"Status:             {res_f['status']}")
    print(f"Sideband Relative:  {res_f['sideband_power_db']:.2f} dB")
    print(f"Total Edge Latency: {res_f['latency']['total_edge_ms']:.3f} ms")
