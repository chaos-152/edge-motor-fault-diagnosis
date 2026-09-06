"""
Cloud Architecture Simulator (AWS EC2 baseline).
Models network latency, data upload payload, and remote compute delay
as documented in Walani & Doorsamy (2025): "Edge vs. Cloud: Empirical Insights into Data-Driven Condition Monitoring".
"""

import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


class AWSCloudSimulator:
    """
    Simulates the cloud-based predictive maintenance workflow:
    1. Raw Stator Current signal batch payload serialization.
    2. Cellular / Industrial Wi-Fi Uplink transmission to AWS EC2 instance.
    3. Remote EC2 processing latency (PSD computation + ML inference).
    4. Downlink transmission of diagnostic result back to plant.
    """

    def __init__(
        self,
        rtt_mean_ms: float = config.CLOUD_SIMULATION["network_rtt_mean_ms"],
        rtt_std_ms: float = config.CLOUD_SIMULATION["network_rtt_std_ms"],
        bandwidth_mbps: float = config.CLOUD_SIMULATION["bandwidth_mbps"],
        ec2_compute_ms: float = config.CLOUD_SIMULATION["ec2_compute_time_ms"]
    ):
        self.rtt_mean_ms = rtt_mean_ms
        self.rtt_std_ms = rtt_std_ms
        self.bandwidth_mbps = bandwidth_mbps
        self.ec2_compute_ms = ec2_compute_ms
        self.rng = np.random.default_rng(config.RANDOM_STATE)

    def calculate_transmission_time_ms(self, payload_bytes: int) -> float:
        """
        Calculates network transmission time based on payload size and bandwidth.
        Transmission time = (Payload in bits) / (Bandwidth in bps) * 1000 ms.
        """
        payload_bits = payload_bytes * 8.0
        bandwidth_bps = self.bandwidth_mbps * 1e6
        tx_time_ms = (payload_bits / bandwidth_bps) * 1000.0
        return tx_time_ms

    def simulate_cloud_inference(
        self,
        signal_array: np.ndarray,
        simulate_network_delay: bool = False
    ) -> Dict[str, float]:
        """
        Simulates end-to-end cloud round trip.

        Args:
            signal_array: Raw time-domain signal array.
            simulate_network_delay: If True, executes time.sleep for true wall-clock delay.

        Returns:
            Dictionary with breakdown of upload time, cloud compute time, total round-trip latency,
            and payload byte count.
        """
        payload_bytes = signal_array.nbytes
        tx_time_ms = self.calculate_transmission_time_ms(payload_bytes)

        # Network RTT with log-normal / Gaussian distribution (industrial jitter)
        network_jitter = self.rng.normal(0, self.rtt_std_ms)
        rtt_latency_ms = max(40.0, self.rtt_mean_ms + network_jitter)

        # Total round-trip latency
        total_latency_ms = tx_time_ms + rtt_latency_ms + self.ec2_compute_ms

        if simulate_network_delay:
            time.sleep(total_latency_ms / 1000.0)

        return {
            "payload_bytes": payload_bytes,
            "payload_kb": payload_bytes / 1024.0,
            "tx_time_ms": tx_time_ms,
            "network_rtt_ms": rtt_latency_ms,
            "ec2_compute_ms": self.ec2_compute_ms,
            "total_round_trip_ms": total_latency_ms
        }
