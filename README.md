# CERT r6.2 Insider Threat Detection — LSTM Autoencoder Project

This project implements an LSTM autoencoder to detect insider threats on the CERT r6.2 synthetic dataset. The pipeline extracts per-user daily behavioral features from five log types, builds fixed-length sequences, trains an autoencoder on benign-only data, and flags anomalies via reconstruction error.

## Overview

- **Goal**: Detect malicious user days (exfiltration/insider threat activity) via unsupervised sequence reconstruction.
- **Dataset**: CERT r6.2 (public insider threat dataset).
- **Approach**:
  - Per-user daily feature extraction (logon, device, email, file, http).
  - Per-user z-score normalization (v2) or global StandardScaler (v1).
  - 30-day sliding-window sequences → LSTM autoencoder.
  - Threshold on reconstruction error (MSE) to classify anomalies.
  - SHAP explanations for feature importance on malicious sequences.

## Project Structure

```
.
├── r6.2/                          # Raw CERT r6.2 logs (logon.csv, device.csv, ...)
├── answers/                       # Ground truth (insiders.csv)
├── processed/                     # Outputs from preprocessing
│   ├── X_train.npy, X_val.npy, X_test.npy
│   ├── y_test.npy, u_test.npy
│   ├── feature_cols.pkl
│   └── malicious_users.pkl
├── models/                        # Trained models and thresholds
│   ├── best_model.keras
│   └── threshold.npy
├── preprocessing.py               # v2 pipeline (richer features + per-user norm)
├── preprocessingv1.py             # v1 baseline (simpler features + global scaler)
├── preprocessingv2.py             # alias/alternate name for v2 (already has docstring)
├── lstm_trainingv1.py             # Baseline LSTM autoencoder training
├── lstm_trainingv2.py             # Improved model (BatchNorm, narrower bottleneck)
├── evaluation.py                  # Test evaluation + ROC/PR/SHAP
├── demo_gui.py                    # Interactive visualization dashboard
├── data_analysis.py               # Week 1: EDA and class imbalance analysis
└── README.md
```

## Quick Start

1. **Place data**
   - Put raw logs (`logon.csv`, `device.csv`, `email.csv`, `file.csv`, `http.csv`) in `r6.2/`.
   - Put `insiders.csv` in `answers/`.

2. **EDA (optional)**
   ```bash
   python data_analysis.py
   ```
   Outputs: `event_frequency.png`, `class_imbalance.png`.

3. **Preprocess**
   - v2 (recommended): richer features + per-user normalization
     ```bash
     python preprocessing.py   # or preprocessingv2.py
     ```
   - v1 (baseline): simpler features + global StandardScaler
     ```bash
     python preprocessingv1.py
     ```
   Outputs (in `processed/`): `X_train.npy`, `X_val.npy`, `X_test.npy`, `y_test.npy`, `u_test.npy`, `feature_cols.pkl`, `malicious_users.pkl` (plus `scaler.pkl` for v1).

4. **Train model**
   - v1 (baseline)
     ```bash
     python lstm_trainingv1.py
     ```
   - v2 (improved)
     ```bash
     python lstm_trainingv2.py
     ```
   Outputs (in `models/`): `best_model.keras`, `threshold.npy`, training plots (`training_curves.png`, `train_reconstruction_errors.png`).

5. **Evaluate**
   ```bash
   python evaluation.py
   ```
   Outputs: `evaluation.png`, `score_distribution.png`, `shap_importance.png`.

6. **Interactive Dashboard**
   ```bash
   python demo_gui.py
   ```
   The dashboard provides an interactive way to explore the model's performance:
   - **Population Scan**: Visualize the entire test population and run a system-wide threat scan.
   - **Sensitivity Adjustment**: Tune the detection threshold in real-time to see impact on True/False Positives.
   - **User Investigation**: Click any user in the grid to see a behavioral breakdown of which features contributed most to their anomaly score.
   - **Metric Tabs**: Switch between detailed views, ROC curves, and training metrics.

## Generated plots

- `event_frequency.png` – Daily event counts by activity type (from data_analysis.py)
- `class_imbalance.png` – Benign vs malicious event distribution (from data_analysis.py)
- `training_curves.png` – Train/val loss curves (from training scripts)
- `train_reconstruction_errors.png` – Reconstruction error distribution on benign data (from training scripts)
- `evaluation.png` – ROC, Precision-Recall, and confusion matrix (from evaluation.py)
- `score_distribution.png` – MSE score distributions for benign vs malicious (from evaluation.py)
- `shap_importance.png` – Feature importances via SHAP for malicious sequences (from evaluation.py)

## Features

Per-day aggregates (per user) extracted across five log types.

- **Logon**: count, after-hours count, mean/std hour (v2 only)
- **Device (USB)**: count, after-hours count (v2 only)
- **Email**: count, total size, attachment count (v2 only)
- **File**: count, after-hours count (v2 only)
- **HTTP**: request count

## Models

### v1 (Baseline)
- Encoder: LSTM(64) → Dropout(0.3) → LSTM(32) bottleneck
- Decoder: RepeatVector → LSTM(32) → Dropout(0.3) → LSTM(64) → TimeDistributed(Dense)
- No BatchNormalization
- Optimizer: Adam(lr=1e-3)
- Threshold: train mean + 2σ

### v2 (Improved)
- Encoder: LSTM(128) → BatchNorm → Dropout(0.2) → LSTM(64) → BatchNorm → Dropout(0.2) → LSTM(16) bottleneck
- Decoder: RepeatVector → LSTM(64) → BatchNorm → Dropout(0.2) → LSTM(128) → BatchNorm → TimeDistributed(Dense)
- Per-user z-score normalization
- Optimizer: Adam(lr=5e-4)
- Threshold: val mean + 3σ (more conservative)
- Early stopping patience = 7, max 50 epochs

## Evaluation Metrics

- ROC-AUC
- Average Precision (AP)
- Precision / Recall / F1
- Confusion matrix
- SHAP feature importances (mean |SHAP|) for malicious sequences

## Notes

- Training uses **benign-only** sequences; validation is also benign-only. The test set contains both benign and malicious user days.
- Per-user normalization (v2) helps detect deviations from an individual's normal pattern, rather than population-level anomalies.
- Threshold selection on the test set is done to maximize F1 for demonstration; in production, tune on a labeled validation set.

## Dependencies

- Python 3.x
- TensorFlow / Keras
- pandas, numpy, scikit-learn
- matplotlib, seaborn
- SHAP

## License

Project code for educational/demonstration purposes using the CERT r6.2 dataset.
