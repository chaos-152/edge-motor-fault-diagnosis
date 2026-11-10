"""
Feature Extraction module for Motor Current Signature Analysis (MCSA).
Extracts edge-viable physical and statistical features from Stator Current and Welch PSD.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

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
        "fault_band_energy_ratio",    # Energy in fault sideband region / low-band energy
        "fundamental_energy_ratio",   # Fundamental carrier energy / low-band energy (scale-invariant)

        # Grid-Frequency-Normalized Spectral Statistical Features
        "spectral_centroid_ratio",    # Spectral center of mass normalized by fundamental frequency
        "spectral_spread_ratio",      # Spectral spread normalized by fundamental frequency
        "spectral_flatness",          # Geometric mean / arithmetic mean of normalized PSD
        "spectral_crest_factor",      # Peak PSD / Mean PSD
        "spectral_entropy",           # Shannon entropy of normalized PSD distribution

        # Per-Unit Stator Current Time-Domain Features
        "load_current_pu",            # Per-unit RMS current (I_rms / I_rated)
        "current_kurtosis",           # Kurtosis of current signal (peakedness)
        "current_crest_factor",       # Peak-to-RMS ratio
        "current_shape_factor"        # RMS / Mean absolute value
    ]

    def __init__(
        self,
        psd_extractor: Optional[WelchPSDExtractor] = None,
        fs: int = config.TARGET_SAMPLING_RATE_HZ,
        supply_freq_hz: Optional[float] = None,
        i_noload_a: float = config.MOTOR_NOLOAD_CURRENT_A,
        i_rated_a: float = config.MOTOR_RATED_CURRENT_A
    ):
        self.psd_extractor = psd_extractor or WelchPSDExtractor(fs=fs)
        self.fs = fs
        if supply_freq_hz is None and not config.SUPPLY_FREQUENCY_AUTO_DETECT:
            supply_freq_hz = config.SUPPLY_FREQUENCY_HZ
        self.supply_freq = supply_freq_hz
        self.i_noload_a = float(i_noload_a)
        self.i_rated_a = float(i_rated_a)

    def extract_from_signal(
        self,
        signal_array: np.ndarray,
        estimated_slip: Optional[float] = None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Extracts feature vector from a raw 1D current signal using single-pass Welch PSD.

        Args:
            signal_array: 1D stator current time series.
            estimated_slip: Motor operating slip. If None, estimated from per-unit load curve.

        Returns:
            feature_vector: 1D numpy array of length len(FEATURE_NAMES).
            feature_dict: Dictionary mapping feature names and telemetry metadata to scalars.
        """
        rms_val = float(np.sqrt(np.mean(signal_array ** 2))) + 1e-12
        sig_pu = signal_array / rms_val

        frequencies, psd_lin = self.psd_extractor.compute_psd(sig_pu, return_db=False)
        psd_db = 10.0 * np.log10(np.maximum(psd_lin, 1e-30))

        return self.extract_from_psd(
            frequencies, psd_db, psd_lin, signal_array=signal_array, estimated_slip=estimated_slip
        )

    def extract_from_psd(
        self,
        frequencies: np.ndarray,
        psd_db: np.ndarray,
        psd_lin: np.ndarray,
        signal_array: Optional[np.ndarray] = None,
        estimated_slip: Optional[float] = None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Extracts scale-invariant & grid-frequency-normalized features from PSD.
        """
        # 1. Locate Fundamental Peak (auto-detects 50 Hz / 60 Hz unless fixed)
        fund_peaks = self.psd_extractor.find_spectral_peaks(
            frequencies, psd_db, fund_target_hz=self.supply_freq, search_radius_hz=2.5
        )
        fund_freq = fund_peaks["fundamental_freq_hz"]
        fund_power_db = fund_peaks["fundamental_power"]

        # 2. Compute RMS & calibrate slip from per-unit load curve
        if signal_array is not None and len(signal_array) > 0:
            rms_val = float(np.sqrt(np.mean(signal_array ** 2)))
            sig_pu = signal_array / max(rms_val, 1e-12)
        else:
            rms_val = self.i_rated_a
            sig_pu = None

        load_pu = float(rms_val / max(self.i_rated_a, 1e-12))

        if estimated_slip is None:
            if rms_val <= self.i_rated_a * 1.15:
                denom = max(self.i_rated_a - self.i_noload_a, 1e-6)
                normalized_load = np.clip((rms_val - self.i_noload_a) / denom, 0.0, 1.0)
            else:
                # Fallback for higher-current synthetic waveforms (1.8 A - 3.8 A)
                normalized_load = np.clip((rms_val - 1.8) / (3.8 - 1.8), 0.0, 1.0)
            estimated_slip = float(0.012 + normalized_load * (0.050 - 0.012))

        # Expected sideband frequencies: fb = fs * (1 +/- 2s)
        expected_f_left = fund_freq * (1.0 - 2.0 * estimated_slip)
        expected_f_right = fund_freq * (1.0 + 2.0 * estimated_slip)

        # 3. Locate Sideband Peaks using local maxima
        def find_best_sideband_peak(f_center: float, half_width: float = 1.8) -> float:
            mask = (frequencies >= f_center - half_width) & (frequencies <= f_center + half_width)
            indices = np.where(mask)[0]
            if len(indices) < 3:
                return fund_power_db - 55.0

            peak_powers = []
            for idx in indices[1:-1]:
                if psd_db[idx] > psd_db[idx - 1] and psd_db[idx] > psd_db[idx + 1]:
                    peak_powers.append(float(psd_db[idx]))

            if peak_powers:
                return max(peak_powers)
            return float(np.median(psd_db[indices]))

        left_peak_db = find_best_sideband_peak(expected_f_left)
        right_peak_db = find_best_sideband_peak(expected_f_right)

        sb_left_rel = left_peak_db - fund_power_db
        sb_right_rel = right_peak_db - fund_power_db
        sb_mean_rel = (sb_left_rel + sb_right_rel) / 2.0
        sb_asym = abs(sb_left_rel - sb_right_rel)

        # 4. Fault Band & Fundamental Energy Ratios (frequency bands relative to detected fs)
        low_band_mask = (frequencies >= 0.15 * fund_freq) & (frequencies <= 2.0 * fund_freq)
        fund_exclusion_mask = (frequencies >= fund_freq - 1.2) & (frequencies <= fund_freq + 1.2)
        fault_sideband_energy_mask = low_band_mask & (~fund_exclusion_mask)

        total_band_energy = float(np.sum(psd_lin[low_band_mask])) + 1e-30
        sideband_energy = float(np.sum(psd_lin[fault_sideband_energy_mask]))
        fund_band_energy = float(np.sum(psd_lin[fund_exclusion_mask]))

        fault_band_energy_ratio = sideband_energy / total_band_energy
        fundamental_energy_ratio = fund_band_energy / total_band_energy

        # 5. Grid-Frequency-Normalized Spectral Statistical Features (0.08*fs to 4.2*fs)
        analysis_mask = (frequencies >= 0.08 * fund_freq) & (frequencies <= 4.2 * fund_freq)
        f_sub_ratio = frequencies[analysis_mask] / max(fund_freq, 1e-6)
        p_raw = psd_lin[analysis_mask]
        p_norm_total = p_raw / (float(np.sum(p_raw)) + 1e-30)
        p_sub = np.maximum(p_norm_total, 1e-12)
        p_norm = p_sub / np.sum(p_sub)

        spectral_centroid_ratio = float(np.sum(f_sub_ratio * p_norm))
        spectral_spread_ratio = float(np.sqrt(np.sum(((f_sub_ratio - spectral_centroid_ratio) ** 2) * p_norm)))

        geo_mean = np.exp(np.mean(np.log(p_sub)))
        arith_mean = np.mean(p_sub)
        spectral_flatness = float(geo_mean / (arith_mean + 1e-12))
        spectral_crest = float(np.max(p_sub) / (arith_mean + 1e-12))

        p_entropy = -np.sum(p_norm * np.log2(p_norm + 1e-12))
        max_entropy = np.log2(len(p_norm))
        spectral_entropy = float(p_entropy / (max_entropy + 1e-12))

        # 6. Scale-Invariant Time-Domain Waveform Shape Statistics (on per-unit signal)
        if sig_pu is not None and len(sig_pu) > 0:
            peak_pu = float(np.max(np.abs(sig_pu)))
            mean_abs_pu = float(np.mean(np.abs(sig_pu))) + 1e-12
            centered = sig_pu - np.mean(sig_pu)
            m2 = float(np.mean(centered ** 2))
            kurt_val = float(np.mean(centered ** 4) / (m2 * m2 + 1e-30) - 3.0)
            crest_factor = peak_pu  # Since RMS of sig_pu is 1.0
            shape_factor = float(1.0 / mean_abs_pu)
        else:
            kurt_val = -1.5
            crest_factor = np.sqrt(2.0)
            shape_factor = 1.11

        feature_dict = {
            "sideband_left_db_rel": float(sb_left_rel),
            "sideband_right_db_rel": float(sb_right_rel),
            "sideband_mean_db_rel": float(sb_mean_rel),
            "sideband_asymmetry_db": float(sb_asym),
            "fault_band_energy_ratio": float(fault_band_energy_ratio),
            "fundamental_energy_ratio": float(fundamental_energy_ratio),
            "spectral_centroid_ratio": float(spectral_centroid_ratio),
            "spectral_spread_ratio": float(spectral_spread_ratio),
            "spectral_flatness": float(spectral_flatness),
            "spectral_crest_factor": float(spectral_crest),
            "spectral_entropy": float(spectral_entropy),
            "load_current_pu": float(load_pu),
            "current_kurtosis": float(kurt_val),
            "current_crest_factor": float(crest_factor),
            "current_shape_factor": float(shape_factor),
            # Human-readable telemetry keys (excluded from ML feature_vector)
            "current_rms_a": float(rms_val),
            "fundamental_freq_hz": float(fund_freq),
            "estimated_slip": float(estimated_slip),
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
