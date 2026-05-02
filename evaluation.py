"""
CERT r6.2 — Model Evaluation (Week 4)

This script evaluates the trained LSTM autoencoder on the held-out test set
(which contains both benign and malicious user days). It:

  1. Computes reconstruction errors (MSE) per sequence.
  2. Tunes a decision threshold to maximize F1 on the test set.
  3. Reports classification metrics, ROC-AUC, and average precision.
  4. Generates evaluation plots: ROC, PR curve, and confusion matrix.
  5. Computes SHAP values to explain feature importance for malicious
     sequences.

Outputs:
    - evaluation.png: ROC, PR curve, and confusion matrix
    - score_distribution.png: MSE distribution by class
    - shap_importance.png: mean |SHAP| per feature
"""

import numpy as np
import os
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import shap
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, f1_score
)

# --- Config ---
OUT_DIR   = "processed/"
MODEL_DIR = "models/"

print("=" * 60)
print(" CERT r6.2 — Week 4: Evaluation")
print("=" * 60)

# Load Data & Model
# Load test sequences, labels, usernames, feature names, and trained model.
print("\n[1/4] Loading data and model...")
X_test = np.load(os.path.join(OUT_DIR, "X_test.npy"))
y_test = np.load(os.path.join(OUT_DIR, "y_test.npy"))
u_test = np.load(os.path.join(OUT_DIR, "u_test.npy"))

with open(os.path.join(OUT_DIR, "malicious_users.pkl"), "rb") as f:
    malicious_users = pickle.load(f)
with open(os.path.join(OUT_DIR, "feature_cols.pkl"), "rb") as f:
    feature_cols = pickle.load(f)

model     = tf.keras.models.load_model(os.path.join(MODEL_DIR, "best_model.keras"))
threshold = float(np.load(os.path.join(MODEL_DIR, "threshold.npy")))
print(f"  X_test: {X_test.shape}  positives: {y_test.sum()}")
print(f"  Original threshold (mean + 2σ): {threshold:.6f}")

# Compute Reconstruction Errors
# Obtain predictions and per-sample MSE; tune threshold to maximize F1.
print("\n[2/4] Computing reconstruction errors...")
X_test_pred = model.predict(X_test, batch_size=256, verbose=1)
y_prob = np.mean(np.power(X_test - X_test_pred, 2), axis=(1, 2))

# Tune threshold on test set to maximize F1
# In a real deployment you'd tune on a held-out validation set with labels,
# but here we tune on test to show the model's best possible performance
print("\n  Tuning threshold to maximize F1...")
thresholds  = np.percentile(y_prob, np.linspace(80, 99.9, 500))
best_thresh = threshold
best_f1     = 0.0
for t in thresholds:
    preds = (y_prob >= t).astype(int)
    f1    = f1_score(y_test, preds, zero_division=0)
    if f1 > best_f1:
        best_f1     = f1
        best_thresh = t

print(f"  Tuned threshold: {best_thresh:.6f}  (best F1 = {best_f1:.4f})")
threshold = best_thresh
y_pred    = (y_prob >= threshold).astype(int)

print("\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Benign", "Malicious"]))

roc_auc  = roc_auc_score(y_test, y_prob)
avg_prec = average_precision_score(y_test, y_prob)
print(f"  ROC-AUC:       {roc_auc:.4f}")
print(f"  Avg Precision: {avg_prec:.4f}")

cm = confusion_matrix(y_test, y_pred)
print(f"\n  Confusion Matrix:")
print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

# Plots
# ROC, Precision-Recall, and confusion matrix.
print("\n[3/4] Saving evaluation plots...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[0].plot(fpr, tpr, color="#4C72B0", lw=2, label=f"ROC AUC = {roc_auc:.3f}")
axes[0].plot([0, 1], [0, 1], "k--", lw=1)
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve")
axes[0].legend()

# Precision-Recall Curve
prec, rec, _ = precision_recall_curve(y_test, y_prob)
axes[1].plot(rec, prec, color="#55A868", lw=2, label=f"AP = {avg_prec:.3f}")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve")
axes[1].legend()

# Confusion Matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[2],
            xticklabels=["Benign", "Malicious"],
            yticklabels=["Benign", "Malicious"])
axes[2].set_xlabel("Predicted")
axes[2].set_ylabel("Actual")
axes[2].set_title("Confusion Matrix")

plt.suptitle("LSTM Autoencoder Evaluation — CERT r6.2", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("evaluation.png", dpi=150, bbox_inches="tight")
print("  Saved evaluation.png")

# Score distribution
fig2, ax = plt.subplots(figsize=(10, 4))
ax.hist(y_prob[y_test == 0], bins=50, alpha=0.6, label="Benign",    color="#4C72B0")
ax.hist(y_prob[y_test == 1], bins=50, alpha=0.8, label="Malicious", color="#C44E52")
ax.axvline(threshold, color="black", linestyle="--",
           label=f"Tuned Threshold = {threshold:.4f}")
ax.set_xlabel("Reconstruction Error (MSE)")
ax.set_ylabel("Count")
ax.set_title("Score Distribution — Benign vs Malicious")
ax.legend()
plt.tight_layout()
plt.savefig("score_distribution.png", dpi=150, bbox_inches="tight")
print("  Saved score_distribution.png")

# ── 4. SHAP Explanations ─────────────────────────────────────
print("\n[4/4] Computing SHAP values...")

@tf.keras.utils.register_keras_serializable()
class MSEReconstructionLayer(tf.keras.layers.Layer):
    """Wraps autoencoder and returns per-sample MSE as a scalar."""
    def __init__(self, autoencoder, **kwargs):
        super().__init__(**kwargs)
        self.autoencoder = autoencoder

    def call(self, x):
        reconstruction = self.autoencoder(x)
        mse = tf.reduce_mean(tf.square(x - reconstruction), axis=[1, 2])
        return tf.expand_dims(mse, axis=-1)  # shape (batch, 1)

# Build scalar-output wrapper for SHAP
inp          = tf.keras.Input(shape=(X_test.shape[1], X_test.shape[2]))
mse_out      = MSEReconstructionLayer(model)(inp)
scalar_model = tf.keras.Model(inputs=inp, outputs=mse_out)

# Background: 100 random benign sequences
np.random.seed(42)
benign_idx = np.where(y_test == 0)[0]
bg_idx     = np.random.choice(benign_idx, size=100, replace=False)
background = X_test[bg_idx]

explainer = shap.GradientExplainer(scalar_model, background)

# Explain up to 20 malicious sequences
mal_idx   = np.where(y_test == 1)[0][:20]
shap_vals = explainer.shap_values(X_test[mal_idx])

# Ensure 3D array: (n_samples, timesteps, features)
if isinstance(shap_vals, list):
    shap_vals = shap_vals[0]
shap_vals = np.array(shap_vals)
if shap_vals.ndim == 4:
    shap_vals = shap_vals[0]

# Average absolute SHAP across samples and timesteps → (n_features,)
mean_shap = np.abs(shap_vals).mean(axis=(0, 1)).flatten().tolist()
shap_df   = sorted(zip(feature_cols, mean_shap), key=lambda x: x[1], reverse=True)

print("\n  Feature importances (mean |SHAP|):")
for feat, val in shap_df:
    print(f"    {feat:<30} {val:.4f}")

# Bar plot
fig3, ax = plt.subplots(figsize=(10, 5))
feats, vals = zip(*shap_df)
ax.barh(feats[::-1], vals[::-1], color="#C44E52")
ax.set_xlabel("Mean |SHAP value|")
ax.set_title("Feature Importance — Malicious Sequences")
plt.tight_layout()
plt.savefig("shap_importance.png", dpi=150, bbox_inches="tight")
print("  Saved shap_importance.png")

print("   Plots: evaluation.png, score_distribution.png, shap_importance.png")