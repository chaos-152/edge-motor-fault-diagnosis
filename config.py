"""
Configuration module for Edge-Based Induction Motor Fault Diagnosis.
Ref: "Edge vs. Cloud: Empirical Insights into Data-Driven Condition Monitoring"
     DOI: 10.3390/bdcc9050121
     "Experimental database for detecting and diagnosing rotor broken bar in a three-phase induction motor"
     IEEE DataPort (Treml et al., 2020)
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / ".cache"
os.environ["MPLCONFIGDIR"] = str(CACHE_DIR / "matplotlib")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models" / "saved_models"
REPORTS_DIR = BASE_DIR / "reports"
PLOTS_DIR = REPORTS_DIR / "plots"

for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, PLOTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Electrical Motor Parameters (Treml et al., 2020 benchmark setup)
SUPPLY_FREQUENCY_HZ = 60.0       # Fundamental electrical frequency (60 Hz grid)
NUM_ROTOR_BARS = 34              # Total squirrel cage rotor bars
PAIRS_OF_POLES = 2               # 4-pole machine (synchronous speed = 1800 RPM at 60 Hz)
RATED_POWER_KW = 0.75            # 1 HP / 0.75 kW induction motor

# Operational Loads & Corresponding Nominal Slips
# Load conditions: 0.5 Nm, 1.0 Nm, 2.0 Nm, 3.0 Nm, 4.0 Nm (Treml et al., 2020)
LOAD_TORQUES_NM = [0.5, 1.0, 2.0, 3.0, 4.0]
NOMINAL_SLIPS = {
    0.5: 0.012,   # ~1778 RPM
    1.0: 0.019,   # ~1765 RPM
    2.0: 0.031,   # ~1744 RPM
    3.0: 0.042,   # ~1724 RPM
    4.0: 0.054    # ~1702 RPM
}

# Signal Acquisition & Sampling Settings
# IEEE DataPort raw sampling is 50 kHz. For edge deployment on Raspberry Pi,
# decimation / target sampling frequency of 2,000 Hz captures all sidebands (0-250 Hz)
# and motor harmonics up to 1,000 Hz with low memory & sub-millisecond computational latency.
RAW_SAMPLING_RATE_HZ = 50000
TARGET_SAMPLING_RATE_HZ = 2000   # Sufficient for MCSA up to 1000 Hz Nyquist
WINDOW_DURATION_SEC = 2.0        # 2.0-second analysis window (4,000 samples)

# Welch Power Spectral Density (PSD) Configuration
# SciPy Welch method: equivalent to MATLAB's pwelch(x, window, noverlap, nfft, fs)
WELCH_WINDOW = "hann"
WELCH_NPERSEG = 2048             # Segment length: frequency resolution ~ 0.98 Hz
WELCH_NOVERLAP = 1024            # 50% overlap
WELCH_NFFT = 4096                # FFT length: interpolated bin resolution ~ 0.49 Hz
WELCH_SCALING = "density"        # V^2/Hz or A^2/Hz (density)

# Motor Current Signature Analysis (MCSA) Feature Extraction Bands
# Broken rotor bar sidebands appear at: f_brb = f_s * (1 +/- 2*k*s)
# Primary sideband harmonic k = 1
SIDEBAND_SEARCH_HALF_WIDTH_HZ = 3.5  # Tolerance window around theoretical sidebands

# Fault Severities
# 0 = Healthy, 1 = 1 broken bar, 2 = 2 broken bars, 3 = 3 broken bars, 4 = 4 broken bars
FAULT_CLASSES = {
    0: "Healthy",
    1: "1 Broken Bar",
    2: "2 Broken Bars",
    3: "3 Broken Bars",
    4: "4 Broken Bars"
}
BINARY_CLASSES = {
    0: "Healthy",
    1: "Faulty (Broken Rotor Bar)"
}

# Machine Learning Configurations
RANDOM_STATE = 42
TEST_SPLIT_RATIO = 0.20
CV_FOLDS = 5

# SVM Hyperparameters
SVM_CONFIG = {
    "C": 10.0,
    "kernel": "rbf",
    "gamma": "scale",
    "random_state": RANDOM_STATE
}

# Decision Tree Hyperparameters (Edge-optimized for near-instant inference)
DECISION_TREE_CONFIG = {
    "max_depth": 6,
    "min_samples_split": 4,
    "min_samples_leaf": 2,
    "criterion": "gini",
    "random_state": RANDOM_STATE
}

# Cloud Simulation Parameters (AWS EC2 baseline from Walani & Doorsamy 2025)
# RTT latency: network transmission time for raw 50kS/s signal batch + remote EC2 compute
CLOUD_SIMULATION = {
    "network_rtt_mean_ms": 160.0,     # Typical 4G/industrial Wi-Fi round-trip to AWS us-east
    "network_rtt_std_ms": 35.0,
    "raw_signal_upload_size_kb": 200.0, # 1 sec of 50kHz float32 ~ 200 KB
    "bandwidth_mbps": 15.0,            # Typical industrial wireless uplink
    "ec2_compute_time_ms": 8.5         # AWS c5.large compute time
}
