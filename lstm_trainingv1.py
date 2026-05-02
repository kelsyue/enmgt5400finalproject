"""
CERT r6.2 — LSTM Autoencoder Training v1 (Baseline)

This script trains a baseline LSTM autoencoder on benign-only sequences
to learn normal behavioral patterns. The model is trained to reconstruct
its input; anomalous sequences (malicious user days) are expected to yield
higher reconstruction errors (MSE).

Architecture (V1):
    - Encoder: LSTM(64, return_sequences=True) → Dropout(0.3)
               → LSTM(32, return_sequences=False) [bottleneck]
    - Decoder: RepeatVector → LSTM(32, return_sequences=True)
               → Dropout(0.3) → LSTM(64, return_sequences=True)
               → TimeDistributed(Dense(n_features))
    - No BatchNormalization
    - Optimizer: Adam(lr=1e-3), Loss: MSE
    - Max 30 epochs with early stopping (patience=5)

Thresholding:
    - Uses training-set reconstruction errors (benign only)
    - Threshold = mean + 2*std of training MSE

Outputs (saved to models/):
    - best_model.keras: best model by validation loss
    - threshold.npy: anomaly threshold (float)
    - training_curves_v1.png: train/val loss curves
    - train_reconstruction_errors_v1.png: benign error distribution
"""

import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, RepeatVector, TimeDistributed
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# --- Config ---
OUT_DIR   = "processed/"
MODEL_DIR = "models/"
os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 60)
print(" CERT r6.2 — Week 3: LSTM Autoencoder Training v1")
print("=" * 60)

# GPU Check
# Report available GPU devices; fallback to CPU if none.
print("\n[GPU Check]")
gpus = tf.config.list_physical_devices("GPU")
print(f"  GPUs available: {len(gpus)}")
for g in gpus:
    print(f"  → {g}")
if not gpus:
    print("  No GPU — training on CPU")

# Load Data
# Load preprocessed benign-only train/val sequences and inspect shapes.
print("\n[1/4] Loading preprocessed data...")
try:
    X_train = np.load(os.path.join(OUT_DIR, "X_train.npy"))
    X_val   = np.load(os.path.join(OUT_DIR, "X_val.npy"))
except FileNotFoundError:
    print(f"ERROR: Run preprocessing.py first.")
    exit(1)

X_train = np.nan_to_num(X_train)
X_val   = np.nan_to_num(X_val)

n_timesteps = X_train.shape[1]
n_features  = X_train.shape[2]
print(f"  X_train: {X_train.shape}  (all benign)")
print(f"  X_val:   {X_val.shape}  (all benign)")
print(f"  Timesteps: {n_timesteps}  Features: {n_features}")

# Build V1 LSTM Autoencoder
# Baseline architecture without BatchNormalization.
# Encoder compresses to 32-dim bottleneck; decoder reconstructs sequence.
print("\n[2/4] Building V1 LSTM Autoencoder...")

inputs = Input(shape=(n_timesteps, n_features))

# Encoder
x       = LSTM(64, return_sequences=True)(inputs)
x       = Dropout(0.3)(x)
encoded = LSTM(32, return_sequences=False)(x)      # 32-dim bottleneck

# Decoder
x       = RepeatVector(n_timesteps)(encoded)
x       = LSTM(32, return_sequences=True)(x)
x       = Dropout(0.3)(x)
x       = LSTM(64, return_sequences=True)(x)
outputs = TimeDistributed(Dense(n_features))(x)

autoencoder = Model(inputs, outputs)
autoencoder.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="mse",
)
autoencoder.summary()

# Train
# Train autoencoder to minimize reconstruction error on benign sequences.
print("\n[3/4] Training...")

callbacks = [
    ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, "best_model.keras"),
        monitor="val_loss", mode="min",
        save_best_only=True, verbose=1,
    ),
    EarlyStopping(
        monitor="val_loss", mode="min",
        patience=5,
        restore_best_weights=True, verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss", mode="min",
        factor=0.5, patience=3, min_lr=1e-6, verbose=1,
    ),
]

history = autoencoder.fit(
    X_train, X_train,
    validation_data=(X_val, X_val),
    epochs=30,                         # V1: max 30 epochs
    batch_size=256,
    callbacks=callbacks,
    verbose=1,
)

# Threshold & Plots
# Compute training-set threshold (mean + 2σ) and save diagnostic plots.
print("\n[4/4] Computing threshold and saving plots...")

# V1: threshold computed from TRAINING set errors (mean + 2σ)
print("  Computing reconstruction errors on training set...")
X_train_pred = autoencoder.predict(X_train, batch_size=256, verbose=0)
train_errors = np.mean(np.power(X_train - X_train_pred, 2), axis=(1, 2))

threshold = float(np.mean(train_errors) + 2 * np.std(train_errors))
print(f"  Anomaly threshold (train mean + 2σ): {threshold:.6f}")
np.save(os.path.join(MODEL_DIR, "threshold.npy"), threshold)

# Training curves
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(history.history["loss"],     label="train loss (MSE)")
ax.plot(history.history["val_loss"], label="val loss (MSE)")
ax.set_title("Autoencoder v1 Training — Reconstruction Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE")
ax.legend()
plt.tight_layout()
plt.savefig("training_curves_v1.png", dpi=150, bbox_inches="tight")
print("  Saved training_curves_v1.png")

# Training error distribution
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(train_errors, bins=100, color="#4C72B0", alpha=0.8)
ax.axvline(threshold, color="#C44E52", linestyle="--", linewidth=2,
           label=f"Threshold = {threshold:.4f}")
ax.set_title("Training Set Reconstruction Error Distribution (Benign Only)")
ax.set_xlabel("MSE")
ax.set_ylabel("Count")
ax.legend()
plt.tight_layout()
plt.savefig("train_reconstruction_errors_v1.png", dpi=150, bbox_inches="tight")
print("  Saved train_reconstruction_errors_v1.png")

print(f"   Best model: {MODEL_DIR}best_model.keras")
print(f"   Threshold:  {threshold:.6f}")