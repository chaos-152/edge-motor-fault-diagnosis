# Edge-Based Fault Diagnosis of Induction Motors: A Lightweight Edge-AI Architecture

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Target: Linux / ARM64](https://img.shields.io/badge/Edge-Linux%20%7C%20ARM64%20Ready-red.svg)](https://github.com/chaos-152/edge-motor-fault-diagnosis)
[![Reference: BDCC 2025](https://img.shields.io/badge/DOI-10.3390%2Fbdcc9050121-green.svg)](https://doi.org/10.3390/bdcc9050121)
[![Dataset: IEEE DataPort](https://img.shields.io/badge/Dataset-IEEE%20DataPort-orange.svg)](https://dx.doi.org/10.21227/fmnm-bn95)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)

**Author:** Sai Samanyu K (`231CS152`)  
**Affiliation:** National Institute of Technology Karnataka (NITK)  
**Keywords:** Edge AI &bull; Predictive Maintenance (PdM) &bull; Industrial IoT (IIoT) &bull; Motor Current Signature Analysis (MCSA) &bull; Zero-Leakage Validation

---

## 1. Overview & Motivation

Industrial condition monitoring of rotating electrical machinery is a cornerstone of Predictive Maintenance (PdM). This project reproduces and adapts the comparative condition monitoring framework established by:

> **Walani, C. C., & Doorsamy, W. (2025).**  
> *Edge vs. Cloud: Empirical Insights into Data-Driven Condition Monitoring.*  
> **Big Data and Cognitive Computing (BDCC)**, MDPI, 9(5), 121. [DOI: 10.3390/bdcc9050121](https://doi.org/10.3390/bdcc9050121).

### From Proprietary Cloud to Portable Open-Source Edge
Traditional cloud architectures stream continuous raw waveforms over wireless links to remote servers, introducing latency (`150–250 ms`), continuous uplink ingestion (`~19.3 GiB/month` single-phase at 2 kHz), recurring cloud/MATLAB license fees, and blind spots during network outages.

This repository implements an **open-source, edge-native Python pipeline** verified locally on Linux x86_64 and designed for portable ARM64 deployment (Android Termux or Raspberry Pi-class boards via NumPy-only execution):
1. **Per-Unit Signal Processing:** Replaces MATLAB `pwelch` with single-pass SciPy/NumPy Welch PSD extraction, auto-detecting `50 Hz` or `60 Hz` fundamentals and normalizing waveforms to per-unit scale.
2. **Active-Motor Data Hygiene:** Filters unpowered motor runs (`RMS < 0.15 A`, e.g., an unenergized trial at 0.5 Nm in `struct_rs_R1.mat`) at both training and inference time to prevent false broken-bar alarms on sensor noise.
3. **Zero-Leakage Evaluation:** Splits train/test strictly by **independent experimental recordings** (holdout repetitions `8–9`, zero shared recordings between splits), plus Leave-One-Torque-Out (LOTO) and perturbation stress tests.
4. **Portable Dependency-Free Runtime:** Exports trained Decision Trees to compact JSON (`1.1 KB`) runnable in `edge/portable_edge.py` using Python + NumPy alone (no `scikit-learn` or `SciPy` required at inference).

```
+---------------------------------------------------------------------------------------------------+
|                                      SYSTEM PIPELINE ARCHITECTURE                                 |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  [IEEE DataPort Recordings]     (Treml et al., 2020: 1,245 active 2.0s windows across 249 runs)   |
|            |                                                                                      |
|            v                                                                                      |
|  [Active Current Gate + Per-Unit Welch PSD]  (Skip RMS < 0.15A; auto-detect 50/60 Hz fundamental) |
|            |                                                                                      |
|            v                                                                                      |
|  [15 Normalized MCSA Features]  (Relative sidebands fs(1 ± 2s), energy ratios, moments, load pu)  |
|            |                                                                                      |
|            v                                                                                      |
|  [Recording-Grouped ML Training] (SVM 6.8 KB & Decision Tree 2.3 KB + 1.1 KB JSON export)         |
|            |                                                                                      |
|            v                                                                                      |
|  [Local Edge Benchmarks & Domain-Shift Audit] (Measured local host vs Simulated cloud RTT model)  |
|                                                                                                   |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. Honest Evaluation & Empirical Results

Evaluated across **1,245 active empirical windows** (excluding 5 dead-motor noise windows from `struct_rs_R1.mat` torque `0.5 Nm` rep `8`) with **strict recording-grouped splitting**: `1,000` training windows (`repetitions 0–7`, 200 recordings) and `245` completely unseen test windows (`repetitions 8–9`, 49 recordings).

### Classification Accuracy (Zero Within-Recording Leakage)

| Task & Split Protocol | Decision Tree (2.3 KB) | SVM RBF Pipeline (6.8–44.5 KB) | Notes |
| :--- | :---: | :---: | :--- |
| **Binary (Healthy vs. Faulty) — 5-Fold GroupKFold** | 98.90% &plusmn; 1.96% | **100.00% &plusmn; 0.00%** | Strict recording-grouped CV |
| **Binary (Healthy vs. Faulty) — Unseen Reps 8&ndash;9** | **99.18%** (`F1: 0.9918`) | **99.59%** (`F1: 0.9959`) | Zero shared 20s recordings |
| **Binary Leave-One-Torque-Out (Unseen Loads)** | **92.64%** mean | 91.96% mean | Per-torque: 81.2%–100.0% (DT) / 80.0%–100.0% (SVM) |
| **Multiclass (0–4 Broken Bars) — 5-Fold GroupKFold** | 90.70% &plusmn; 2.18% | **99.40% &plusmn; 0.97%** | SVM strongly outperforms DT on bar count |
| **Multiclass (0–4 Broken Bars) — Unseen Reps 8&ndash;9** | 81.22% | **98.78%** (`F1: 0.9878`) | Adjacent fault severities overlap in shallow trees |
| **Multiclass Leave-One-Torque-Out (Unseen Loads)** | 37.18% mean | 59.68% mean | Bar-fault amplitude couples strongly with load torque |

### Domain-Shift & Perturbation Stress Audit (Unseen Reps 8&ndash;9)

| Scenario | Decision Tree Binary Acc | SVM Binary Acc | Engineering Interpretation |
| :--- | :---: | :---: | :--- |
| Clean Unseen Recordings (`reps 8–9`) | **99.18%** | **99.59%** | Reference baseline |
| Calibrated Sensor Gain &times;2.5 (matched `I_rated`) | **99.18%** | **99.59%** | Exact per-unit calibration invariance |
| Uncalibrated Sensor Gain &times;1.5 / &times;0.5 | **99.18%** / **99.18%** | 81.63% / 80.41% | DT unaffected; SVM uses `load_current_pu` |
| Additive Gaussian Noise (&sigma; = 0.05 A / 0.08 A) | 81.63% / 80.00% | 81.63% / 81.63% | Degrades smoothly under high broadband noise |
| Synthetic 50 Hz Cross-Grid Test (`n=2`) | 50.00% | 50.00% | 50 Hz carrier & sideband detected, but classifier learned 60 Hz motor harmonic profile |

### Latency, Resource Footprint & Cloud Baseline Comparison

* **Measurement Transparency:** Edge latency and RAM usage below were measured directly on the development Linux host (`x86_64`, Python 3.13) plus single-core CPU affinity pinning (`os.sched_setaffinity`). **No live AWS EC2 network call or physical Raspberry Pi execution was run**—the AWS EC2 comparison column is a **simulated baseline** computed via `AWSCloudSimulator` using literature network parameters (`160 ± 35 ms` WAN RTT, `15 Mbps` uplink, `8.5 ms` server compute) from Walani & Doorsamy (2025).

| Metric | Measured Local Edge (DT) | Single-Core Pinned Edge (DT) | Measured Local Edge (SVM) | Simulated Cloud Baseline (*Walani 2025 Model*) |
| :--- | :---: | :---: | :---: | :---: |
| **Mean Latency (150 runs)** | **0.844 ms** (`p95: 1.21 ms`) | **0.736 ms** (`p95: 0.87 ms`) | **0.938 ms** (`p95: 1.46 ms`) | 175.263 ms (`p95: 221.30 ms`) [Simulated] |
| **Throughput** | **1,012.5 windows/sec** | &mdash; | &mdash; | Network RTT-bound (~5.7 windows/sec) |
| **Memory Leak (300 cycles)** | **0.00 MB growth** (`225.7 MB` RSS) | &mdash; | &mdash; | Server-side |
| **Continuous 2 kHz Bandwidth** | **0.625 KiB/day** (alarms only) | &mdash; | **0.625 KiB/day** | **659.2 MiB/day** (~19.31 GiB/month, 1 phase) |

---

## 3. Repository Structure

```
.
├── config.py                      # Motor specs, nameplate calibration, DSP & zero-leakage split config
├── run_pipeline.py                # Master CLI: ingestion, training, robustness audit, benchmarks & dashboard
├── data/
│   ├── dataset_loader.py          # IEEE DataPort HDF5 parser (with active-current gate) & synthetic MCSA generator
│   ├── samples/                   # Lightweight (<150 KB) unseen IEEE sample bundle for portable ARM64 runs
│   └── raw/ / processed/          # Local IEEE DataPort .mat archives and extracted .npz/.csv caches
├── dsp/
│   ├── welch_psd.py               # Single-pass Welch PSD (SciPy + bit-exact NumPy fallback, 50/60 Hz auto-detect)
│   └── feature_extractor.py       # 15 per-unit, grid-frequency-normalized MCSA & statistical features
├── models/
│   ├── train.py                   # Recording-grouped SVM & DT training, LOTO validation & JSON tree export
│   ├── evaluate.py                # Classification metrics & domain-shift perturbation audit (robustness_metrics.json)
│   └── saved_models/              # Exported .joblib models and 1.1 KB portable JSON trees
├── edge/
│   ├── edge_runtime.py            # Offline single-pass diagnostic engine with standby current gate & CLI demo
│   ├── portable_edge.py           # Zero-sklearn, NumPy-only ARM64/Termux edge inference runner
│   ├── cloud_simulator.py         # Literature-parameterized WAN/AWS EC2 delay simulator
│   ├── benchmark.py               # Latency, Linux single-core affinity, psutil RSS & bandwidth benchmarking
│   └── deploy_pi.sh               # Optional ARM64 Raspberry Pi OS Bookworm/Trixie deployment & systemd setup
├── visualization/
│   ├── plot_signals.py            # Generates 300-DPI matched-torque waveform, PSD, CM & benchmark charts
│   └── dashboard.py               # Standalone interactive HTML report generator
└── tests/
    └── test_pipeline.py           # Automated pytest suite (DSP parity, gain invariance, zero leakage, inference)
```

---

## 4. Quickstart & Portable Execution

### Option A: Standard Local Environment (Linux / macOS / Windows)
```bash
git clone https://github.com/chaos-152/edge-motor-fault-diagnosis.git
cd edge-motor-fault-diagnosis

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run automated tests (zero-leakage split, gain invariance, standby gate, NumPy parity)
pytest tests/ -v

# Run full end-to-end training, robustness audit, benchmarks, and dashboard generation
python run_pipeline.py

# Inspect MCSA sideband physics across all load torques (0.5 to 4.0 Nm)
python edge/edge_runtime.py --compare
```

### Option B: Portable ARM64 / Android Termux Execution (NumPy Only)
Because `dsp/welch_psd.py` and `edge/portable_edge.py` run with **Python + NumPy alone** and use the bundled `143 KB` unseen sample file (`data/samples/ieee_validation_samples.npz`), you can execute genuine ARM64 offline inference on an Android phone using Termux (or any slim Linux board) without installing `scikit-learn` or downloading the 7.3 GB dataset:

```bash
pkg install python python-numpy git
git clone https://github.com/chaos-152/edge-motor-fault-diagnosis.git
cd edge-motor-fault-diagnosis
python edge/portable_edge.py --stream
```

### Option C: Docker Containerization
```bash
docker compose up --build
```

---

## 5. Known Scope & Limitations

1. **Single-Motor Benchmark Validation:** All empirical training uses Treml et al.'s `1 HP` (0.75 kW), 4-pole, 34-bar WEG induction motor. Deploying to motors with different horsepower, bar counts, or saturation harmonics requires updating `MOTOR_NOLOAD_CURRENT_A` / `MOTOR_RATED_CURRENT_A` in `config.py` and retraining or recalibrating against healthy baseline runs on the target machine.
2. **Multiclass Cross-Torque Limits:** Broken rotor bar current modulation decreases sharply near zero load (`0.5 Nm`), making *exact bar count* (`1` vs `2` vs `3` vs `4` bars) strongly dependent on load torque (`59.7%` SVM LOTO mean vs `98.8%` in-distribution unseen recordings). In field deployments lacking shaft torque telemetry, binary fault detection is markedly more reliable than unseen-load severity counting.
3. **Simulated Cloud & Host Hardware:** Local CPU latency (`~0.8 ms` on 13th-Gen x86_64) proves algorithm computational feasibility under sub-2 ms single-core execution budgets, while the `175.3 ms` cloud baseline represents a literature-parameterized WAN simulation rather than live EC2 instrumentation.

---

## 6. Benchmark Dataset & Citations

* **Experimental Dataset:** Treml, A. E., Flauzino, R. A., Suetake, M., & Maciejewski, N. A. R. (2020). *Experimental database for detecting and diagnosing rotor broken bar in a three-phase induction motor*. IEEE DataPort. [DOI: 10.21227/fmnm-bn95](https://dx.doi.org/10.21227/fmnm-bn95).
* **Reference Framework:** Walani, C. C., & Doorsamy, W. (2025). *Edge vs. Cloud: Empirical Insights into Data-Driven Condition Monitoring*. Big Data and Cognitive Computing, 9(5), 121. [DOI: 10.3390/bdcc9050121](https://doi.org/10.3390/bdcc9050121).

---

## License
Distributed under the MIT License. See `LICENSE` for more information.
