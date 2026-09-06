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


def test_cloud_simulator():
    """Verifies cloud round-trip delay modeling."""
    sim = AWSCloudSimulator()
    dummy_signal = np.zeros(4000, dtype=np.float32)
    res = sim.simulate_cloud_inference(dummy_signal, simulate_network_delay=False)

    assert res["payload_bytes"] == 4000 * 4
    assert res["total_round_trip_ms"] > 50.0 # RTT is significantly higher than edge latency
