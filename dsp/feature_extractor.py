"""
Feature Extraction module for Motor Current Signature Analysis (MCSA).
Extracts edge-viable physical and statistical features from Stator Current and Welch PSD.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from dsp.welch_psd import WelchPSDExtractor


class MCSAFeatureExtractor:
    """
    Extracts physical MCSA sideband signatures and spectral/time-domain features.
    """

    FEATURE_NAMES = [
        # Physical MCSA Sideband Features (Broken Rotor Bar Signatures)
        "sideband_left_db_rel",       # Left sideband power relative to fundamental (dB)
        "sideband_right_db_rel",      # Right sideband power relative to fundamental (dB)
        "sideband_mean_db_rel",       # Mean sideband power relative to fundamental (dB)
        "sideband_asymmetry_db",      # Difference between left and right sideband power (dB)
        "fault_band_energy_ratio",    # Energy in fault sideband region / total band energy
        "fundamental_power_db",       # Absolute power of fundamental (dB/Hz)

        # Spectral Domain Statistical Features
        "spectral_centroid_hz",       # Center of mass of the spectrum
        "spectral_spread_hz",         # Standard deviation of spectrum frequencies
        "spectral_flatness",          # Geometric mean / arithmetic mean of power
        "spectral_crest_factor",      # Peak PSD / Mean PSD
        "spectral_entropy",           # Shannon entropy of normalized PSD distribution

        # Time-Domain Stator Current Features
        "current_rms_a",              # Root Mean Square current (Amperes)
        "current_kurtosis",           # Kurtosis of current signal (peakedness)
        "current_crest_factor",       # Peak-to-RMS ratio
        "current_shape_factor"        # RMS / Mean absolute deviation
    ]

    def __init__(
        self,
        psd_extractor: Optional[WelchPSDExtractor] = None,
        fs: int = config.TARGET_SAMPLING_RATE_HZ,
        supply_freq_hz: float = config.SUPPLY_FREQUENCY_HZ
    ):
        self.psd_extractor = psd_extractor or WelchPSDExtractor(fs=fs)
        self.fs = fs
        self.supply_freq = supply_freq_hz

    def extract_from_signal(
        self,
        signal_array: np.ndarray,
        estimated_slip: Optional[float] = None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Extracts feature vector from a raw 1D current signal.

        Args:
            signal_array: 1D stator current time series.
            estimated_slip: Motor operating slip. If None, estimated from nominal range.

        Returns:
            feature_vector: 1D numpy array of length len(FEATURE_NAMES).
            feature_dict: Dictionary mapping feature name to scalar value.
        """
        frequencies, psd_db = self.psd_extractor.compute_psd(signal_array, return_db=True)
        # Also compute linear PSD for energy integrals and spectral moments
        _, psd_lin = self.psd_extractor.compute_psd(signal_array, return_db=False)

        return self.extract_from_psd(frequencies, psd_db, psd_lin, signal_array, estimated_slip)

    def extract_from_psd(
        self,
        frequencies: np.ndarray,
        psd_db: np.ndarray,
        psd_lin: np.ndarray,
        signal_array: Optional[np.ndarray] = None,
        estimated_slip: Optional[float] = None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Extracts features when PSD is already computed.
        """
        # 1. Locate Fundamental Peak (fs ~ 60 Hz)
        fund_peaks = self.psd_extractor.find_spectral_peaks(
            frequencies, psd_db, fund_target_hz=self.supply_freq, search_radius_hz=2.5
        )
        fund_freq = fund_peaks["fundamental_freq_hz"]
        fund_power_db = fund_peaks["fundamental_power"]

        # 2. Estimate slip from stator current RMS if not provided
        # Induction motor current scales monotonically with mechanical load and slip
        if signal_array is not None and len(signal_array) > 0:
            rms_val = float(np.sqrt(np.mean(signal_array ** 2)))
        else:
            rms_val = 2.5

        if estimated_slip is None:
            # Linear interpolation based on nominal motor curves:
            # 1.8 A RMS -> slip ~0.012 (0.5 Nm), 3.8 A RMS -> slip ~0.054 (4.0 Nm)
            normalized_load = np.clip((rms_val - 1.8) / (3.8 - 1.8), 0.0, 1.0)
            estimated_slip = float(0.012 + normalized_load * (0.054 - 0.012))

        # Expected sideband frequencies: fb = fs * (1 +/- 2s)
        expected_f_left = fund_freq * (1.0 - 2.0 * estimated_slip)
        expected_f_right = fund_freq * (1.0 + 2.0 * estimated_slip)

        # 3. Locate Sideband Peaks using local maxima
        def find_best_sideband_peak(f_center: float, half_width: float = 1.8) -> float:
            mask = (frequencies >= f_center - half_width) & (frequencies <= f_center + half_width)
            indices = np.where(mask)[0]
            if len(indices) < 3:
                return fund_power_db - 55.0

            # Find local peaks
            peak_powers = []
            for idx in indices[1:-1]:
                if psd_db[idx] > psd_db[idx - 1] and psd_db[idx] > psd_db[idx + 1]:
                    peak_powers.append(float(psd_db[idx]))

            if peak_powers:
                return max(peak_powers)
            else:
                # If no distinct local peak, take the median of the search zone
                return float(np.median(psd_db[indices]))

        left_peak_db = find_best_sideband_peak(expected_f_left)
        right_peak_db = find_best_sideband_peak(expected_f_right)

        sb_left_rel = left_peak_db - fund_power_db
        sb_right_rel = right_peak_db - fund_power_db
        sb_mean_rel = (sb_left_rel + sb_right_rel) / 2.0
        sb_asym = abs(sb_left_rel - sb_right_rel)

        # 4. Fault Band Energy Ratio (10 Hz to 120 Hz zone)
        # Sideband power zone vs total band power
        low_band_mask = (frequencies >= 10.0) & (frequencies <= 120.0)
        fund_exclusion_mask = (frequencies >= fund_freq - 1.2) & (frequencies <= fund_freq + 1.2)
        fault_sideband_energy_mask = low_band_mask & (~fund_exclusion_mask)

        total_band_energy = float(np.sum(psd_lin[low_band_mask])) + 1e-12
        sideband_energy = float(np.sum(psd_lin[fault_sideband_energy_mask]))
        fault_band_energy_ratio = sideband_energy / total_band_energy

        # 4. Spectral Domain Statistical Features (calculated in 0-250 Hz range)
        analysis_mask = (frequencies >= 5.0) & (frequencies <= 250.0)
        f_sub = frequencies[analysis_mask]
        p_sub = np.maximum(psd_lin[analysis_mask], 1e-12)
        p_norm = p_sub / (np.sum(p_sub) + 1e-12)

        # Spectral Centroid
        spectral_centroid = float(np.sum(f_sub * p_norm))

        # Spectral Spread
        spectral_spread = float(np.sqrt(np.sum(((f_sub - spectral_centroid) ** 2) * p_norm)))

        # Spectral Flatness (Geometric Mean / Arithmetic Mean)
        # Using exp(mean(log)) to avoid underflow
        geo_mean = np.exp(np.mean(np.log(p_sub)))
        arith_mean = np.mean(p_sub)
        spectral_flatness = float(geo_mean / (arith_mean + 1e-12))

        # Spectral Crest Factor (Peak PSD / Mean PSD)
        spectral_crest = float(np.max(p_sub) / (arith_mean + 1e-12))

        # Spectral Entropy
        p_entropy = -np.sum(p_norm * np.log2(p_norm + 1e-12))
        max_entropy = np.log2(len(p_norm))
        spectral_entropy = float(p_entropy / (max_entropy + 1e-12))

        # 5. Time-Domain Features
        if signal_array is not None and len(signal_array) > 0:
            rms_val = float(np.sqrt(np.mean(signal_array ** 2)))
            peak_val = float(np.max(np.abs(signal_array)))
            mean_abs = float(np.mean(np.abs(signal_array))) + 1e-12
            kurt_val = float(stats.kurtosis(signal_array))
            crest_factor = float(peak_val / (rms_val + 1e-12))
            shape_factor = float(rms_val / mean_abs)
        else:
            rms_val = 2.5
            kurt_val = -1.5
            crest_factor = np.sqrt(2.0)
            shape_factor = 1.11

        feature_dict = {
            "sideband_left_db_rel": float(sb_left_rel),
            "sideband_right_db_rel": float(sb_right_rel),
            "sideband_mean_db_rel": float(sb_mean_rel),
            "sideband_asymmetry_db": float(sb_asym),
            "fault_band_energy_ratio": float(fault_band_energy_ratio),
            "fundamental_power_db": float(fund_power_db),
            "spectral_centroid_hz": float(spectral_centroid),
            "spectral_spread_hz": float(spectral_spread),
            "spectral_flatness": float(spectral_flatness),
            "spectral_crest_factor": float(spectral_crest),
            "spectral_entropy": float(spectral_entropy),
            "current_rms_a": float(rms_val),
            "current_kurtosis": float(kurt_val),
            "current_crest_factor": float(crest_factor),
            "current_shape_factor": float(shape_factor)
        }

        feature_vector = np.array([feature_dict[name] for name in self.FEATURE_NAMES], dtype=np.float32)
        return feature_vector, feature_dict

    def extract_batch(
        self,
        signals: np.ndarray,
        slips: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Extracts feature vectors for a batch of signals of shape (N, samples).
        """
        n_samples = signals.shape[0]
        feature_matrix = np.zeros((n_samples, len(self.FEATURE_NAMES)), dtype=np.float32)

        for i in range(n_samples):
            slip_i = slips[i] if slips is not None else None
            feat_vec, _ = self.extract_from_signal(signals[i], estimated_slip=slip_i)
            feature_matrix[i] = feat_vec

        return feature_matrix


if __name__ == "__main__":
    from data.dataset_loader import StatorCurrentDataset
    dataset = StatorCurrentDataset()
    extractor = MCSAFeatureExtractor()

    sig_h, _ = dataset.generate_single_signal(fault_severity=0, load_torque_nm=2.0)
    sig_f, _ = dataset.generate_single_signal(fault_severity=2, load_torque_nm=2.0)

    vec_h, dict_h = extractor.extract_from_signal(sig_h)
    vec_f, dict_f = extractor.extract_from_signal(sig_f)

    print(f"Extracted {len(vec_h)} features per sample.")
    print(f"Healthy Sideband Mean Relative: {dict_h['sideband_mean_db_rel']:.2f} dB")
    print(f"Faulty (2 BRB) Sideband Mean Relative: {dict_f['sideband_mean_db_rel']:.2f} dB")
    print(f"Healthy Fault Band Energy Ratio: {dict_h['fault_band_energy_ratio']:.4f}")
    print(f"Faulty Fault Band Energy Ratio: {dict_f['fault_band_energy_ratio']:.4f}")
