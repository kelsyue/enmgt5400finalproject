"""
CERT r6.2 — LSTM Autoencoder Training v2 (Improved)

This script trains an improved LSTM autoencoder with stronger regularization
and a narrower bottleneck to better capture normal patterns and increase
reconstruction error separation for anomalies.

Improvements over v1:
    - Added BatchNormalization after each LSTM layer for stable training
    - Wider encoder (128 → 64) to capture complex patterns
    - Narrower bottleneck (16 instead of 32) to force stronger compression
    - Lower learning rate (5e-4) for careful convergence
    - More epochs (50) with increased early-stopping patience (7)
    - Threshold computed from validation set (mean + 3σ) for better
      generalization and fewer false positives

Architecture:
    - Encoder: LSTM(128) → BatchNorm → Dropout(0.2)
               → LSTM(64) → BatchNorm → Dropout(0.2)
               → LSTM(16) [bottleneck]
    - Decoder: RepeatVector → LSTM(64) → BatchNorm → Dropout(0.2)
               → LSTM(128) → BatchNorm → TimeDistributed(Dense(n_features))

Outputs (saved to models/):
    - best_model.keras: best model by validation loss
    - threshold.npy: anomaly threshold (float)
    - training_curves.png: train/val loss curves
    - train_reconstruction_errors.png: validation error distribution
"""

import numpy as np
import os
import shutil
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, RepeatVector,
    TimeDistributed, BatchNormalization
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# --- Config ---
OUT_DIR   = "processed/"
MODEL_DIR = "models/"
os.makedirs(MODEL_DIR, exist_ok=True)

# Backup old model before overwriting
old_model  = os.path.join(MODEL_DIR, "best_model.keras")
old_thresh = os.path.join(MODEL_DIR, "threshold.npy")
if os.path.exists(old_model):
    shutil.copy(old_model,  os.path.join(MODEL_DIR, "best_model_v1_backup.keras"))
    print("  ✅ Backed up → models/best_model_v1_backup.keras")
if os.path.exists(old_thresh):
    shutil.copy(old_thresh, os.path.join(MODEL_DIR, "threshold_v1_backup.npy"))
    print("  ✅ Backed up → models/threshold_v1_backup.npy")

print("=" * 60)
print(" CERT r6.2 — Week 3: LSTM Autoencoder Training v2")
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
print(f"  X_train: {X_train.shape}")
print(f"  X_val:   {X_val.shape}")
print(f"  Timesteps: {n_timesteps}  Features: {n_features}")

# Build Improved LSTM Autoencoder
# Stronger regularization and narrower bottleneck to capture normal patterns.
print("\n[2/4] Building improved LSTM Autoencoder...")

inputs = Input(shape=(n_timesteps, n_features))

# Encoder — compress sequence into a bottleneck representation
x = LSTM(128, return_sequences=True)(inputs)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
x = LSTM(64, return_sequences=True)(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
encoded = LSTM(16, return_sequences=False)(x)   # bottleneck: 16-dim

# Decoder — reconstruct original sequence from bottleneck
x = RepeatVector(n_timesteps)(encoded)
x = LSTM(64, return_sequences=True)(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
x = LSTM(128, return_sequences=True)(x)
x = BatchNormalization()(x)
outputs = TimeDistributed(Dense(n_features))(x)

autoencoder = Model(inputs, outputs)
autoencoder.compile(
    optimizer=Adam(learning_rate=5e-4),   # lower lr = more careful convergence
    loss="mse",
)
autoencoder.summary()

# Train
# Train autoencoder with early stopping and learning-rate reduction.
print("\n[3/4] Training...")

callbacks = [
    ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, "best_model.keras"),
        monitor="val_loss", mode="min",
        save_best_only=True, verbose=1,
    ),
    EarlyStopping(
        monitor="val_loss", mode="min",
        patience=7,                        # more patience than v1
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
    epochs=50,                             # more epochs, early stopping will catch plateau
    batch_size=256,
    callbacks=callbacks,
    verbose=1,
)

# Threshold & Plots
# Compute validation-set threshold (mean + 3σ) and save diagnostic plots.
print("\n[4/4] Computing threshold and saving plots...")

# CHANGE FROM V1: use VALIDATION errors to set threshold
# Val set is benign-only but unseen during training — more reliable than train set
print("  Computing reconstruction errors on validation set...")
X_val_pred = autoencoder.predict(X_val, batch_size=256, verbose=0)
val_errors = np.mean(np.power(X_val - X_val_pred, 2), axis=(1, 2))

# mean + 3σ is more conservative — fewer false positives
threshold = float(np.mean(val_errors) + 3 * np.std(val_errors))
print(f"  Anomaly threshold (val mean + 3σ): {threshold:.6f}")
np.save(os.path.join(MODEL_DIR, "threshold.npy"), threshold)

# Training curves
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(history.history["loss"],     label="train loss (MSE)")
ax.plot(history.history["val_loss"], label="val loss (MSE)")
ax.set_title("Autoencoder v2 Training — Reconstruction Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE")
ax.legend()
plt.tight_layout()
plt.savefig("training_curves.png", dpi=150, bbox_inches="tight")
print("  Saved training_curves.png")

# Validation error distribution with threshold
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(val_errors, bins=100, color="#4C72B0", alpha=0.8)
ax.axvline(threshold, color="#C44E52", linestyle="--", linewidth=2,
           label=f"Threshold = {threshold:.4f}")
ax.set_title("Validation Set Reconstruction Error (Benign Only)")
ax.set_xlabel("MSE")
ax.set_ylabel("Count")
ax.legend()
plt.tight_layout()
plt.savefig("train_reconstruction_errors.png", dpi=150, bbox_inches="tight")
print("  Saved train_reconstruction_errors.png")

print(f"   Best model: {MODEL_DIR}best_model.keras")
print(f"   Threshold:  {threshold:.6f}")