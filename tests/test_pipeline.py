"""
Automated unit and integration test suite for the Induction Motor Fault Diagnosis pipeline.
"""

import os
import sys
from pathlib import Path

import pytest
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from data.dataset_loader import StatorCurrentDataset
from dsp.welch_psd import WelchPSDExtractor
from dsp.feature_extractor import MCSAFeatureExtractor
from models.train import ModelTrainer
from edge.edge_runtime import EdgeFaultClassifier
from edge.portable_edge import PortableDecisionTreeNode, diagnose_portable
from edge.cloud_simulator import AWSCloudSimulator


@pytest.fixture
def dataset():
    return StatorCurrentDataset()


@pytest.fixture
def psd_extractor():
    return WelchPSDExtractor()


@pytest.fixture
def feature_extractor(psd_extractor):
    return MCSAFeatureExtractor(psd_extractor=psd_extractor)


def test_data_generation(dataset):
    """Verifies physical signal generation and parameter ranges."""
    sig_healthy, meta_h = dataset.generate_single_signal(fault_severity=0, load_torque_nm=2.0)
    sig_faulty, meta_f = dataset.generate_single_signal(fault_severity=3, load_torque_nm=2.0)

    expected_samples = int(config.WINDOW_DURATION_SEC * config.TARGET_SAMPLING_RATE_HZ)
    assert len(sig_healthy) == expected_samples
    assert len(sig_faulty) == expected_samples
    assert not np.isnan(sig_healthy).any()
    assert not np.isnan(sig_faulty).any()

    # Fundamental RMS check (at 2.0 Nm, RMS is ~2.4 to 3.0 A)
    rms_h = np.sqrt(np.mean(sig_healthy ** 2))
    assert 1.5 < rms_h < 4.5

    # Sideband frequency check
    assert meta_f["f_left_hz"] < meta_f["supply_freq_hz"]
    assert meta_f["f_right_hz"] > meta_f["supply_freq_hz"]


def test_welch_psd_extraction(psd_extractor):
    """Verifies that Welch's method detects the 60 Hz fundamental accurately."""
    t = np.linspace(0, config.WINDOW_DURATION_SEC, int(config.TARGET_SAMPLING_RATE_HZ * config.WINDOW_DURATION_SEC), endpoint=False)
    test_signal = 3.0 * np.sin(2 * np.pi * 60.0 * t) + 0.05 * np.random.randn(len(t))

    f, psd_db = psd_extractor.compute_psd(test_signal, return_db=True)
    assert len(f) == len(psd_db)
    assert f[0] == 0.0
    assert f[-1] == config.TARGET_SAMPLING_RATE_HZ / 2.0

    peaks = psd_extractor.find_spectral_peaks(f, psd_db, fund_target_hz=60.0)
    assert abs(peaks["fundamental_freq_hz"] - 60.0) < 1.0
    assert peaks["fundamental_power"] > 0.0


def test_mcsa_feature_extraction(feature_extractor, dataset):
    """Verifies feature extraction output shapes, names, and physical sideband elevation."""
    sig_h, _ = dataset.generate_single_signal(fault_severity=0, load_torque_nm=2.0)
    sig_f, _ = dataset.generate_single_signal(fault_severity=3, load_torque_nm=2.0)

    vec_h, dict_h = feature_extractor.extract_from_signal(sig_h)
    vec_f, dict_f = feature_extractor.extract_from_signal(sig_f)

    assert len(vec_h) == len(MCSAFeatureExtractor.FEATURE_NAMES)
    assert len(vec_f) == len(MCSAFeatureExtractor.FEATURE_NAMES)
    assert not np.isnan(vec_h).any()
    assert not np.isnan(vec_f).any()

    # Physical property: sideband power is elevated for 3 broken rotor bars
    assert dict_f["sideband_mean_db_rel"] > dict_h["sideband_mean_db_rel"]
    assert dict_f["fault_band_energy_ratio"] > dict_h["fault_band_energy_ratio"]


def test_edge_runtime_inference(dataset):
    """Verifies that the edge diagnostic engine runs under latency budget with correct output."""
    classifier = EdgeFaultClassifier(model_type="decision_tree", target_type="binary")

    processed_file = config.PROCESSED_DATA_DIR / "stator_current_dataset.npz"
    sample_file = config.SAMPLES_DIR / "ieee_validation_samples.npz"
    if processed_file.exists():
        signals, _, labels_bin, meta = dataset.load_processed_dataset()
        idx_h = int(next(i for i, m in enumerate(meta) if m["fault_severity"] == 0 and m.get("repetition") in config.HOLDOUT_REPETITIONS))
        idx_f = int(next(i for i, m in enumerate(meta) if m["fault_severity"] >= 3 and m.get("repetition") in config.HOLDOUT_REPETITIONS))
        sig_healthy, sig_faulty = signals[idx_h], signals[idx_f]
    elif sample_file.exists():
        pack = np.load(sample_file, allow_pickle=True)
        idx_h = int(np.where(pack["labels_bin"] == 0)[0][0])
        idx_f = int(np.where(pack["labels_bin"] == 1)[0][0])
        sig_healthy, sig_faulty = pack["signals"][idx_h], pack["signals"][idx_f]
    else:
        sig_healthy, _ = dataset.generate_single_signal(fault_severity=0, load_torque_nm=2.0)
        sig_faulty, _ = dataset.generate_single_signal(fault_severity=3, load_torque_nm=2.0)

    res_h = classifier.diagnose_waveform(sig_healthy)
    res_f = classifier.diagnose_waveform(sig_faulty)

    assert res_h["status"] == "Healthy"
    assert res_h["is_faulty"] is False

    assert res_f["status"] == "Faulty (Broken Rotor Bar)"
    assert res_f["is_faulty"] is True

    # Edge Latency constraint: sub-50ms budget for edge execution
    assert res_h["latency"]["total_edge_ms"] < 50.0
    assert res_f["latency"]["total_edge_ms"] < 50.0


def test_numpy_welch_parity_and_50hz_autodetect(psd_extractor):
    """Verifies pure-NumPy Welch matches SciPy and auto-detects both 50 Hz and 60 Hz grids."""
    t = np.linspace(0, config.WINDOW_DURATION_SEC, int(config.TARGET_SAMPLING_RATE_HZ * config.WINDOW_DURATION_SEC), endpoint=False)
    sig_50 = 2.0 * np.sin(2 * np.pi * 50.0 * t) + 0.02 * np.random.default_rng(1).normal(size=len(t))

    f_sp, p_sp = psd_extractor.compute_psd(sig_50, return_db=False)
    f_np, p_np = psd_extractor._numpy_welch(sig_50)
    assert np.allclose(f_sp, f_np)
    assert np.max(np.abs(p_sp - p_np) / (p_sp + 1e-12)) < 1e-10

    peaks_50 = psd_extractor.find_spectral_peaks(f_sp, 10.0 * np.log10(np.maximum(p_sp, 1e-30)))
    assert abs(peaks_50["fundamental_freq_hz"] - 50.0) < 1.0


def test_per_unit_gain_invariance(dataset):
    """Verifies calibrated sensor gain scaling produces invariant feature vectors."""
    sig, _ = dataset.generate_single_signal(fault_severity=2, load_torque_nm=3.0, rng=np.random.default_rng(7))
    ext_base = MCSAFeatureExtractor()
    ext_scaled = MCSAFeatureExtractor(
        i_noload_a=config.MOTOR_NOLOAD_CURRENT_A * 3.0,
        i_rated_a=config.MOTOR_RATED_CURRENT_A * 3.0,
    )
    vec_base, _ = ext_base.extract_from_signal(sig, estimated_slip=0.04)
    vec_scaled, _ = ext_scaled.extract_from_signal(sig * 3.0, estimated_slip=0.04)
    assert np.allclose(vec_base, vec_scaled, atol=1e-4)


def test_motor_off_standby_gate():
    """Verifies near-zero current waveforms trigger Standby instead of false broken-bar alarms."""
    classifier = EdgeFaultClassifier(model_type="decision_tree", target_type="binary")
    noise_only = np.random.default_rng(0).normal(0.0, 0.004, size=4000).astype(np.float32)
    res = classifier.diagnose_waveform(noise_only)
    assert res["status"] == "Motor Off / Standby"
    assert res["is_faulty"] is False
    assert res["prediction_class"] == -1


def test_recording_grouped_split_zero_leakage():
    """Verifies train/test split never places windows from the same recording in both sets."""
    import pandas as pd
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 15)).astype(np.float32)
    y = np.array(([0] * 50) + ([1] * 50))
    reps = np.tile(np.repeat(np.arange(10), 5), 2)
    rec_ids = np.array([f"{y[i]}_{reps[i]}" for i in range(100)])
    df_meta = pd.DataFrame({"repetition": reps, "recording_id": rec_ids})

    _, _, _, _, g_train, g_test, mode = ModelTrainer._split_by_recording(X, y, df_meta)
    assert len(set(g_train).intersection(set(g_test))) == 0
    assert "independent_repetitions" in mode


def test_portable_tree_parity(dataset):
    """Verifies JSON portable Decision Tree matches scikit-learn EdgeFaultClassifier outputs."""
    sklearn_clf = EdgeFaultClassifier(model_type="decision_tree", target_type="binary")
    portable_tree = PortableDecisionTreeNode(config.MODELS_DIR / "decision_tree_portable.json")
    ext = MCSAFeatureExtractor()

    sample_file = config.SAMPLES_DIR / "ieee_validation_samples.npz"
    if sample_file.exists():
        pack = np.load(sample_file, allow_pickle=True)
        signals = pack["signals"]
    else:
        sig_h, _ = dataset.generate_single_signal(fault_severity=0, load_torque_nm=2.0)
        sig_f, _ = dataset.generate_single_signal(fault_severity=3, load_torque_nm=3.0)
        signals = [sig_h, sig_f]

    for sig in signals:
        r_sk = sklearn_clf.diagnose_waveform(sig)
        r_pt = diagnose_portable(sig, portable_tree, ext, target_type="binary")
        assert r_sk["prediction_class"] == r_pt["prediction_class"]


def test_cloud_simulator():
    """Verifies cloud round-trip delay modeling."""
    sim = AWSCloudSimulator()
    dummy_signal = np.zeros(4000, dtype=np.float32)
    res = sim.simulate_cloud_inference(dummy_signal, simulate_network_delay=False)

    assert res["payload_bytes"] == 4000 * 4
    assert res["total_round_trip_ms"] > 50.0 # RTT is significantly higher than edge latency
