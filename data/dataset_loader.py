"""
Dataset loader and high-fidelity MCSA signal generator.
Supports:
1. Raw IEEE DataPort (Treml et al., 2020) .mat / .csv files.
2. High-fidelity synthetic stator current signal generation adhering to Treml et al.
   experimental setup (3-phase squirrel-cage induction motor, 34 rotor bars, 60 Hz fundamental,
   variable mechanical load torque 0.5 - 4.0 Nm, and 0 to 4 broken rotor bars).
"""

import os
import sys
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

import numpy as np
import pandas as pd
from scipy import io as sio


class StatorCurrentDataset:
    """
    Dataset management for induction motor stator current signals.
    """

    def __init__(self, raw_dir: Path = config.RAW_DATA_DIR, processed_dir: Path = config.PROCESSED_DATA_DIR):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)

    def load_ieee_dataport_dataset(
        self,
        target_sampling_rate: int = config.TARGET_SAMPLING_RATE_HZ,
        window_duration_sec: float = config.WINDOW_DURATION_SEC,
        windows_per_repetition: int = 5,
        torques: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        Loads and processes real empirical data from the IEEE DataPort Treml et al. (2020) dataset.
        Parses MATLAB v7.3 HDF5 files:
          - struct_rs_R1.mat  -> Healthy (0 broken bars)
          - struct_r1b_R1.mat -> 1 Broken Bar
          - struct_r2b_R1.mat -> 2 Broken Bars
          - struct_r3b_R1.mat -> 3 Broken Bars
          - struct_r4b_R1.mat -> 4 Broken Bars
        """
        import h5py
        from scipy import signal as dsp_signal

        file_mapping = [
            ("struct_rs_R1.mat", "rs", 0),
            ("struct_r1b_R1.mat", "r1b", 1),
            ("struct_r2b_R1.mat", "r2b", 2),
            ("struct_r3b_R1.mat", "r3b", 3),
            ("struct_r4b_R1.mat", "r4b", 4),
        ]

        if torques is None:
            # Main torque operating points in Treml et al.
            torques = ["torque05", "torque10", "torque20", "torque30", "torque40"]

        torque_values = {
            "torque05": 0.5, "torque10": 1.0, "torque15": 1.5, "torque20": 2.0,
            "torque25": 2.5, "torque30": 3.0, "torque35": 3.5, "torque40": 4.0
        }

        signals = []
        labels_multi = []
        labels_bin = []
        metadata_list = []

        decimate_factor = int(config.RAW_SAMPLING_RATE_HZ / target_sampling_rate) # e.g. 50000 / 2000 = 25
        samples_per_window = int(window_duration_sec * target_sampling_rate)      # e.g. 2.0 * 2000 = 4000

        print(f"Loading real IEEE DataPort dataset from {self.raw_dir}...")

        for filename, group_key, fault_level in file_mapping:
            mat_path = self.raw_dir / filename
            if not mat_path.exists():
                print(f"  [Skip] {filename} not found in {self.raw_dir}")
                continue

            print(f"  -> Reading {filename} (Fault Level {fault_level}: {config.FAULT_CLASSES[fault_level]})...")
            with h5py.File(mat_path, "r") as f:
                if group_key not in f:
                    print(f"     Group '{group_key}' not found in {filename}, skipping.")
                    continue

                root_grp = f[group_key]
                for t_name in torques:
                    if t_name not in root_grp:
                        continue

                    t_grp = root_grp[t_name]
                    torque_nm = torque_values.get(t_name, 2.0)
                    if "Ia" not in t_grp:
                        continue

                    ia_refs = t_grp["Ia"]
                    num_reps = min(len(ia_refs), 10) # 10 experimental repetitions

                    for rep_idx in range(num_reps):
                        ref = ia_refs[rep_idx, 0]
                        # 1,001,000 samples at 50 kHz (~20 seconds)
                        raw_data = np.array(f[ref], dtype=np.float32).flatten()

                        # Skip first 4 seconds (startup transient: 200,000 samples at 50 kHz)
                        # Extract steady-state portion from second 4 to 20
                        steady_data = raw_data[200000:]
                        if len(steady_data) < 100000:
                            continue

                        # Decimate to target sampling rate (e.g. 2000 Hz)
                        decimated = dsp_signal.decimate(steady_data, decimate_factor)

                        # Slice non-overlapping analysis windows
                        max_windows = min(windows_per_repetition, len(decimated) // samples_per_window)
                        for w_idx in range(max_windows):
                            start_i = w_idx * samples_per_window
                            end_i = start_i + samples_per_window
                            win_signal = decimated[start_i:end_i]

                            if len(win_signal) == samples_per_window:
                                signals.append(win_signal.astype(np.float32))
                                labels_multi.append(fault_level)
                                labels_bin.append(int(fault_level > 0))
                                metadata_list.append({
                                    "source": "IEEE_DataPort",
                                    "file": filename,
                                    "fault_severity": fault_level,
                                    "is_faulty": int(fault_level > 0),
                                    "load_torque_nm": torque_nm,
                                    "repetition": rep_idx,
                                    "window": w_idx,
                                    "sampling_rate": target_sampling_rate,
                                    "duration_sec": window_duration_sec
                                })

        if not signals:
            raise RuntimeError(f"No valid IEEE DataPort signals could be extracted from {self.raw_dir}.")

        signals = np.array(signals, dtype=np.float32)
        labels_multi = np.array(labels_multi, dtype=np.int64)
        labels_bin = np.array(labels_bin, dtype=np.int64)

        print(f"Successfully loaded {len(signals)} empirical windows from IEEE DataPort!")
        print(f"  - Healthy samples: {np.sum(labels_bin == 0)}")
        print(f"  - Faulty samples:  {np.sum(labels_bin == 1)}")

        return signals, labels_multi, labels_bin, metadata_list

    def generate_single_signal(
        self,
        fault_severity: int,
        load_torque_nm: float,
        duration_sec: float = config.WINDOW_DURATION_SEC,
        sampling_rate: int = config.TARGET_SAMPLING_RATE_HZ,
        snr_db: float = 38.0,
        rng: Optional[np.random.Generator] = None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Generates a physically accurate stator current waveform based on Motor Current
        Signature Analysis (MCSA) theory and Treml et al. motor parameters.

        Motor Parameters:
        - Fundamental frequency: fs = 60 Hz
        - Number of rotor bars: Nr = 34
        - Load torque: 0.5 to 4.0 Nm
        - Slip: s(load)
        - BRB sideband frequencies: f_brb = fs * (1 +/- 2*k*s), k=1, 2

        Returns:
            signal: 1D numpy array of stator current in Amperes.
            metadata: dict with slip, fundamental frequency, fault severity, load.
        """
        if rng is None:
            rng = np.random.default_rng()

        num_samples = int(duration_sec * sampling_rate)
        t = np.linspace(0, duration_sec, num_samples, endpoint=False)

        # Base electrical supply frequency with tiny grid fluctuation (+/- 0.05 Hz)
        fs = config.SUPPLY_FREQUENCY_HZ + rng.normal(0, 0.03)

        # Nominal slip based on load torque + small mechanical speed jitter
        nominal_slip = config.NOMINAL_SLIPS.get(load_torque_nm, 0.03)
        actual_slip = nominal_slip * (1.0 + rng.normal(0, 0.02))

        # Fundamental stator current amplitude scales with load torque
        # 1 HP motor: ~1.8 A RMS at no load (0.5 Nm) to ~3.8 A RMS at full load (4.0 Nm)
        i_fundamental_peak = 2.2 + 0.65 * load_torque_nm
        phase_offset = rng.uniform(0, 2 * np.pi)

        # 1. Fundamental supply component (60 Hz)
        signal = i_fundamental_peak * np.sin(2 * np.pi * fs * t + phase_offset)

        # 2. Inherent small harmonics present in industrial grids (3rd, 5th, 7th)
        signal += (0.015 * i_fundamental_peak) * np.sin(2 * np.pi * 3 * fs * t)
        signal += (0.010 * i_fundamental_peak) * np.sin(2 * np.pi * 5 * fs * t)

        # 3. Broken Rotor Bar (BRB) Fault Signatures
        # Fault severity: 0 = Healthy, 1 = 1 BRB, 2 = 2 BRB, 3 = 3 BRB, 4 = 4 BRB
        if fault_severity == 0:
            # Healthy rotor has small residual inherent asymmetry (-52 dB to -48 dB)
            sideband_ratio = rng.uniform(0.002, 0.004)
        else:
            # Each broken bar increases the rotating backwards magnetic field
            # In literature, 1 broken bar produces sidebands approx -35 dB (0.018)
            # 4 broken bars produce sidebands approx -18 dB (0.125)
            # Proportional scaling with (n_broken / Nr):
            base_ratio = (fault_severity / config.NUM_ROTOR_BARS) * 1.15
            sideband_ratio = base_ratio * rng.uniform(0.92, 1.08)

        # Primary sideband frequencies (k = 1)
        f_left_1 = fs * (1.0 - 2.0 * actual_slip)
        f_right_1 = fs * (1.0 + 2.0 * actual_slip)

        amp_sideband = i_fundamental_peak * sideband_ratio
        signal += amp_sideband * np.sin(2 * np.pi * f_left_1 * t + rng.uniform(0, 2*np.pi))
        signal += (amp_sideband * 0.85) * np.sin(2 * np.pi * f_right_1 * t + rng.uniform(0, 2*np.pi))

        # Secondary sidebands (k = 2: fs * (1 +/- 4s)) - pronounced for severe faults
        if fault_severity >= 2:
            f_left_2 = fs * (1.0 - 4.0 * actual_slip)
            f_right_2 = fs * (1.0 + 4.0 * actual_slip)
            amp_sideband_2 = amp_sideband * (0.28 * (fault_severity / 2.0))
            signal += amp_sideband_2 * np.sin(2 * np.pi * f_left_2 * t + rng.uniform(0, 2*np.pi))
            signal += amp_sideband_2 * np.sin(2 * np.pi * f_right_2 * t + rng.uniform(0, 2*np.pi))

        # 4. Additive Gaussian White Noise (AWGN) to simulate measurement & sensor noise
        signal_power = np.mean(signal ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10.0))
        noise = rng.normal(0, np.sqrt(noise_power), num_samples)
        signal = signal + noise

        metadata = {
            "fault_severity": fault_severity,
            "is_faulty": int(fault_severity > 0),
            "load_torque_nm": float(load_torque_nm),
            "slip": float(actual_slip),
            "supply_freq_hz": float(fs),
            "f_left_hz": float(f_left_1),
            "f_right_hz": float(f_right_1),
            "sampling_rate": sampling_rate,
            "duration_sec": duration_sec
        }

        return signal.astype(np.float32), metadata

    def generate_benchmark_dataset(
        self,
        samples_per_condition: int = 40,
        random_seed: int = config.RANDOM_STATE
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        Generates a comprehensive dataset covering all 5 fault severities
        (0, 1, 2, 3, 4 broken bars) across all 5 loading conditions (0.5 to 4.0 Nm).

        Total samples = 5 fault levels * 5 loads * samples_per_condition = 1,000 samples.

        Returns:
            signals: (N, num_samples) array
            labels_multiclass: (N,) array with 0..4
            labels_binary: (N,) array with 0 or 1
            metadata_list: list of dicts
        """
        rng = np.random.default_rng(random_seed)
        signals = []
        labels_multi = []
        labels_bin = []
        metadata_list = []

        for fault_level in sorted(config.FAULT_CLASSES.keys()):
            for load_nm in config.LOAD_TORQUES_NM:
                for _ in range(samples_per_condition):
                    sig, meta = self.generate_single_signal(
                        fault_severity=fault_level,
                        load_torque_nm=load_nm,
                        sampling_rate=config.TARGET_SAMPLING_RATE_HZ,
                        rng=rng
                    )
                    signals.append(sig)
                    labels_multi.append(fault_level)
                    labels_bin.append(int(fault_level > 0))
                    metadata_list.append(meta)

        signals = np.array(signals, dtype=np.float32)
        labels_multi = np.array(labels_multi, dtype=np.int64)
        labels_bin = np.array(labels_bin, dtype=np.int64)

        return signals, labels_multi, labels_bin, metadata_list

    def save_dataset(
        self,
        signals: np.ndarray,
        labels_multi: np.ndarray,
        labels_bin: np.ndarray,
        metadata_list: List[Dict],
        filename: str = "stator_current_dataset.npz"
    ) -> Path:
        """Saves generated or parsed dataset to compressed numpy file."""
        out_path = self.processed_dir / filename
        np.savez_compressed(
            out_path,
            signals=signals,
            labels_multi=labels_multi,
            labels_bin=labels_bin,
            metadata=np.array(metadata_list, dtype=object)
        )
        return out_path

    def load_processed_dataset(
        self,
        filename: str = "stator_current_dataset.npz"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """Loads processed dataset from disk."""
        target_path = self.processed_dir / filename
        if not target_path.exists():
            raise FileNotFoundError(f"Processed dataset not found at {target_path}")

        data = np.load(target_path, allow_pickle=True)
        signals = data["signals"]
        labels_multi = data["labels_multi"]
        labels_bin = data["labels_bin"]
        metadata_list = list(data["metadata"])
        return signals, labels_multi, labels_bin, metadata_list


if __name__ == "__main__":
    loader = StatorCurrentDataset()
    print("Testing MCSA signal generator...")
    sig_healthy, meta_h = loader.generate_single_signal(fault_severity=0, load_torque_nm=2.0)
    sig_faulty, meta_f = loader.generate_single_signal(fault_severity=2, load_torque_nm=2.0)
    print(f"Generated healthy signal: shape {sig_healthy.shape}, RMS {np.sqrt(np.mean(sig_healthy**2)):.3f} A")
    print(f"Generated faulty signal: shape {sig_faulty.shape}, RMS {np.sqrt(np.mean(sig_faulty**2)):.3f} A")
    print("Sidebands for faulty: Left =", meta_f["f_left_hz"], "Hz, Right =", meta_f["f_right_hz"], "Hz")
