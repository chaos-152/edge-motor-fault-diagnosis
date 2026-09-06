"""
Power Spectral Density (PSD) extraction using SciPy Welch's method.
Reconstructs MATLAB's pwelch(x, window, noverlap, nfft, fs) for open-source Python edge deployment.
Ref: "Edge vs. Cloud: Empirical Insights into Data-Driven Condition Monitoring" (Walani & Doorsamy, 2025)
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


class WelchPSDExtractor:
    """
    Computes Welch's Power Spectral Density matching MATLAB pwelch behavior.
    """

    def __init__(
        self,
        fs: int = config.TARGET_SAMPLING_RATE_HZ,
        window: str = config.WELCH_WINDOW,
        nperseg: int = config.WELCH_NPERSEG,
        noverlap: int = config.WELCH_NOVERLAP,
        nfft: int = config.WELCH_NFFT,
        scaling: str = config.WELCH_SCALING
    ):
        self.fs = fs
        self.window = window
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nfft
        self.scaling = scaling

    def compute_psd(
        self,
        x: np.ndarray,
        return_db: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the one-sided PSD of a 1D signal.

        Args:
            x: 1D time-domain signal array.
            return_db: If True, returns PSD in dB/Hz: 10 * log10(Pxx + eps).

        Returns:
            frequencies: 1D array of frequency bins in Hz.
            psd: 1D array of PSD values (linear or dB).
        """
        if x.ndim != 1:
            x = np.squeeze(x)

        # scipy.signal.welch: detrend='constant' matches MATLAB pwelch default mean removal
        frequencies, pxx = signal.welch(
            x,
            fs=self.fs,
            window=self.window,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            nfft=self.nfft,
            scaling=self.scaling,
            detrend="constant",
            return_onesided=True
        )

        if return_db:
            # Add small epsilon to avoid log(0)
            eps = 1e-12
            pxx_db = 10.0 * np.log10(np.maximum(pxx, eps))
            return frequencies, pxx_db

        return frequencies, pxx

    def compute_batch_psd(
        self,
        signals: np.ndarray,
        return_db: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes PSD for a batch of signals of shape (N, samples).

        Returns:
            frequencies: (num_freq_bins,) array
            psd_matrix: (N, num_freq_bins) array
        """
        num_signals = signals.shape[0]
        # Compute first to determine output size
        f, sample_psd = self.compute_psd(signals[0], return_db=return_db)
        psd_matrix = np.zeros((num_signals, len(sample_psd)), dtype=np.float32)
        psd_matrix[0] = sample_psd

        for i in range(1, num_signals):
            _, pxx = self.compute_psd(signals[i], return_db=return_db)
            psd_matrix[i] = pxx

        return f, psd_matrix

    def find_spectral_peaks(
        self,
        frequencies: np.ndarray,
        psd: np.ndarray,
        fund_target_hz: float = config.SUPPLY_FREQUENCY_HZ,
        search_radius_hz: float = 3.0
    ) -> Dict[str, float]:
        """
        Locates the exact fundamental peak in the PSD.
        """
        fund_mask = (frequencies >= fund_target_hz - search_radius_hz) & (frequencies <= fund_target_hz + search_radius_hz)
        if not np.any(fund_mask):
            fund_idx = np.argmin(np.abs(frequencies - fund_target_hz))
        else:
            fund_indices = np.where(fund_mask)[0]
            fund_idx = fund_indices[np.argmax(psd[fund_indices])]

        actual_fund_freq = float(frequencies[fund_idx])
        fund_power = float(psd[fund_idx])

        return {
            "fundamental_freq_hz": actual_fund_freq,
            "fundamental_power": fund_power,
            "fundamental_idx": int(fund_idx)
        }


if __name__ == "__main__":
    extractor = WelchPSDExtractor()
    t = np.linspace(0, 1.0, 5000, endpoint=False)
    test_sig = np.sin(2 * np.pi * 60.0 * t) + 0.1 * np.sin(2 * np.pi * 56.4 * t)
    f, psd_db = extractor.compute_psd(test_sig, return_db=True)
    peaks = extractor.find_spectral_peaks(f, psd_db, fund_target_hz=60.0)
    print(f"Computed Welch PSD: {len(f)} frequency bins (0 to {f[-1]} Hz)")
    print(f"Identified fundamental: {peaks['fundamental_freq_hz']:.2f} Hz at {peaks['fundamental_power']:.2f} dB/Hz")
