"""
CERT r6.2 Insider Threat Dataset — Preprocessing Pipeline (v2)

This script processes raw CERT r6.2 log data (logon, device, email, file, http)
into per-user, per-day feature vectors suitable for sequence-based anomaly
detection. It applies per-user normalization and builds fixed-length sliding
window sequences for LSTM autoencoder training.

Key Features:
    - Logon: count, after-hours logins, hour statistics
    - Device: USB activity count, after-hours device usage
    - Email: count, total size, attachment count
    - File: copy count, after-hours file operations
    - HTTP: web request count

Processing Steps:
    1. Load ground truth labels (malicious user windows)
    2. Extract per-user daily aggregates from each activity type
    3. Merge into a complete user-day grid (fill missing with 0)
    4. Apply per-user z-score normalization per feature
    5. Label days falling within malicious incident windows
    6. Split users into train/val/test and build 30-day sequences

Outputs (saved to processed/):
    - X_train.npy, X_val.npy, X_test.npy: (n_samples, 30, n_features)
    - y_test.npy: binary labels for test sequences
    - u_test.npy: usernames for test sequences
    - feature_cols.pkl: list of feature names
    - malicious_users.pkl: set of malicious usernames
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import pickle

# --- Config ---
DATA_DIR    = "r6.2/"
ANSWERS_DIR = "answers/"
OUT_DIR     = "processed/"
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print(" CERT r6.2 — Week 2: Preprocessing v2 (Richer Features)")
print("=" * 60)

# Load Labels
# Load ground truth: confirmed malicious users and their incident windows.
print("\n[1/6] Loading ground truth...")
insiders   = pd.read_csv(os.path.join(ANSWERS_DIR, "insiders.csv"))
r62_labels = insiders[insiders["dataset"] == 6.2].copy()
r62_labels["start"] = pd.to_datetime(r62_labels["start"], format="mixed", errors="coerce")
r62_labels["end"]   = pd.to_datetime(r62_labels["end"],   format="mixed", errors="coerce")
malicious_users = set(r62_labels["user"].unique())
print(f"  Malicious users ({len(malicious_users)}): {sorted(malicious_users)}")

# Build Daily Feature Vectors
# Extract per-activity aggregates (logon, device, email, file, http) with richer features.
# NEW: includes after-hours flags, hour statistics, attachment counts, etc.
print("\n[2/6] Building daily feature vectors (chunked)...")

def process_chunks(filename, agg_funcs, date_col="date"):
    path   = os.path.join(DATA_DIR, filename)
    chunks = []
    for chunk in pd.read_csv(path, chunksize=10**6, parse_dates=[date_col]):
        chunk["day"]  = chunk[date_col].dt.normalize()
        chunk["hour"] = chunk[date_col].dt.hour
        chunk_agg = chunk.groupby(["user", "day"]).agg(**agg_funcs).reset_index()
        chunks.append(chunk_agg)
    full = pd.concat(chunks)
    return full.groupby(["user", "day"]).sum().reset_index()

# --- Logon features ---
# NEW: logon_hour_std captures variance in login times (unusual hours = anomaly)
print("  Processing logon.csv...")
logon_chunks = []
for chunk in pd.read_csv(os.path.join(DATA_DIR, "logon.csv"),
                         chunksize=10**6, parse_dates=["date"]):
    chunk["day"]  = chunk["date"].dt.normalize()
    chunk["hour"] = chunk["date"].dt.hour
    chunk["after_hours"] = ((chunk["hour"] < 8) | (chunk["hour"] >= 18)).astype(int)
    agg = chunk.groupby(["user", "day"]).agg(
        logon_count        = ("activity",    "count"),
        logon_after_hours  = ("after_hours", "sum"),
        logon_hour_mean    = ("hour",        "mean"),
        logon_hour_std     = ("hour",        "std"),
    ).reset_index()
    logon_chunks.append(agg)
logon_feat = pd.concat(logon_chunks).groupby(["user", "day"]).agg({
    "logon_count":       "sum",
    "logon_after_hours": "sum",
    "logon_hour_mean":   "mean",
    "logon_hour_std":    "mean",
}).reset_index().fillna(0)

# --- Device (USB) features ---
# NEW: device_after_hours — USB use at night is a key exfiltration signal
print("  Processing device.csv...")
device_chunks = []
for chunk in pd.read_csv(os.path.join(DATA_DIR, "device.csv"),
                         chunksize=10**6, parse_dates=["date"]):
    chunk["day"]  = chunk["date"].dt.normalize()
    chunk["hour"] = chunk["date"].dt.hour
    chunk["after_hours"] = ((chunk["hour"] < 8) | (chunk["hour"] >= 18)).astype(int)
    agg = chunk.groupby(["user", "day"]).agg(
        device_count       = ("activity",    "count"),
        device_after_hours = ("after_hours", "sum"),
    ).reset_index()
    device_chunks.append(agg)
device_feat = pd.concat(device_chunks).groupby(["user", "day"]).sum().reset_index()

# --- Email features ---
# NEW: attachment_count — large email attachments suggest data exfiltration
print("  Processing email.csv...")
email_chunks = []
for chunk in pd.read_csv(os.path.join(DATA_DIR, "email.csv"),
                         chunksize=10**6, parse_dates=["date"]):
    chunk["day"] = chunk["date"].dt.normalize()
    chunk["attachments"] = pd.to_numeric(chunk["attachments"], errors="coerce").fillna(0)
    agg = chunk.groupby(["user", "day"]).agg(
        email_count       = ("id",          "count"),
        email_size_total  = ("size",        "sum"),
        email_attachments = ("attachments", "sum"),
    ).reset_index()
    email_chunks.append(agg)
email_feat = pd.concat(email_chunks).groupby(["user", "day"]).sum().reset_index()

# --- File copy features ---
# NEW: file_after_hours — copying files to USB after hours is a key red flag
print("  Processing file.csv...")
file_chunks = []
for chunk in pd.read_csv(os.path.join(DATA_DIR, "file.csv"),
                         chunksize=10**6, parse_dates=["date"]):
    chunk["day"]  = chunk["date"].dt.normalize()
    chunk["hour"] = chunk["date"].dt.hour
    chunk["after_hours"] = ((chunk["hour"] < 8) | (chunk["hour"] >= 18)).astype(int)
    agg = chunk.groupby(["user", "day"]).agg(
        file_count       = ("id",          "count"),
        file_after_hours = ("after_hours", "sum"),
    ).reset_index()
    file_chunks.append(agg)
file_feat = pd.concat(file_chunks).groupby(["user", "day"]).sum().reset_index()

# --- HTTP features ---
print("  Processing http.csv (this may take a while)...")
http_feat = process_chunks("http.csv", {
    "http_count": ("id", "count"),
})

# Merge into User-Day Grid
# Combine all activity features into a complete user-day matrix and fill missing entries with 0.
print("\n[3/6] Merging features into user-day grid...")
all_feats = [logon_feat, device_feat, email_feat, file_feat, http_feat]
all_users = set().union(*[set(df["user"]) for df in all_feats])

all_min_date = min(df["day"].min() for df in all_feats)
all_max_date = max(df["day"].max() for df in all_feats)
all_days = pd.date_range(all_min_date, all_max_date, freq="D")

grid  = pd.MultiIndex.from_product([sorted(all_users), all_days], names=["user", "day"])
daily = pd.DataFrame(index=grid).reset_index()

for df in all_feats:
    daily = daily.merge(df, on=["user", "day"], how="left")
daily = daily.fillna(0)

feature_cols = [c for c in daily.columns if c not in ["user", "day", "label"]]
print(f"  Daily feature matrix shape: {daily.shape}")
print(f"  Features ({len(feature_cols)}): {feature_cols}")

# Per-User Normalization
# Normalize each feature relative to that user's own history (z-score).
# NEW: makes the model sensitive to individual behavioral changes rather than
# population-level anomalies.
print("\n[4/6] Applying per-user normalization...")
for col in feature_cols:
    user_mean = daily.groupby("user")[col].transform("mean")
    user_std  = daily.groupby("user")[col].transform("std").replace(0, 1)
    daily[col] = (daily[col] - user_mean) / user_std
daily[feature_cols] = daily[feature_cols].fillna(0)
print("  Done.")

# Add Precise Labels
# Mark days that fall within any malicious incident window.
print("\n[5/6] Adding labels based on incident windows...")
daily["label"] = 0
for _, row in r62_labels.iterrows():
    if pd.isnull(row["start"]) or pd.isnull(row["end"]):
        print(f"  Warning: Skipping malformed date range for user {row['user']}")
        continue
    mask = (
        (daily["user"] == row["user"]) &
        (daily["day"] >= row["start"].normalize()) &
        (daily["day"] <= row["end"].normalize())
    )
    daily.loc[mask, "label"] = 1

print(f"  Benign rows:    {(daily['label']==0).sum():>10,}")
print(f"  Malicious rows: {(daily['label']==1).sum():>10,}")

# Split & Build Sequences
# Split users into train/val/test (by user, not rows) and build 30-day sliding-window sequences.
# Note: per-user normalization already applied so no StandardScaler needed.
print("\n[6/6] Splitting and building sequences...")

benign_users = sorted(all_users - malicious_users)
np.random.seed(42)
np.random.shuffle(benign_users)

n       = len(benign_users)
n_train = int(n * 0.70)
n_val   = int(n * 0.15)

train_users = set(benign_users[:n_train])
val_users   = set(benign_users[n_train:n_train + n_val])
test_users  = set(benign_users[n_train + n_val:]) | malicious_users

# Train/val: benign days only
train_df = daily[(daily["user"].isin(train_users)) & (daily["label"] == 0)].copy()
val_df   = daily[(daily["user"].isin(val_users))   & (daily["label"] == 0)].copy()
test_df  = daily[daily["user"].isin(test_users)].copy()

def build_sequences(df, feature_cols, seq_len=30):
    X_list, y_list, user_list = [], [], []
    for user, grp in df.sort_values("day").groupby("user"):
        feats  = grp[feature_cols].values.astype(np.float32)
        labels = grp["label"].values
        if len(feats) <= seq_len:
            continue
        for i in range(len(feats) - seq_len):
            X_list.append(feats[i:i + seq_len])
            y_list.append(labels[i + seq_len])
            user_list.append(user)
    return np.array(X_list), np.array(y_list), np.array(user_list)

SEQ_LEN = 30
print(f"  Building sequences (window={SEQ_LEN} days)...")
X_train, _,      _      = build_sequences(train_df, feature_cols, SEQ_LEN)
X_val,   _,      _      = build_sequences(val_df,   feature_cols, SEQ_LEN)
X_test,  y_test, u_test = build_sequences(test_df,  feature_cols, SEQ_LEN)

X_train = np.nan_to_num(X_train)
X_val   = np.nan_to_num(X_val)
X_test  = np.nan_to_num(X_test)

print(f"  X_train: {X_train.shape}  (all benign)")
print(f"  X_val:   {X_val.shape}  (all benign)")
print(f"  X_test:  {X_test.shape}  malicious days: {y_test.sum()}")

# Save
np.save(os.path.join(OUT_DIR, "X_train.npy"), X_train)
np.save(os.path.join(OUT_DIR, "X_val.npy"),   X_val)
np.save(os.path.join(OUT_DIR, "X_test.npy"),  X_test)
np.save(os.path.join(OUT_DIR, "y_test.npy"),  y_test)
np.save(os.path.join(OUT_DIR, "u_test.npy"),  u_test)
with open(os.path.join(OUT_DIR, "feature_cols.pkl"),    "wb") as f: pickle.dump(feature_cols,    f)
with open(os.path.join(OUT_DIR, "malicious_users.pkl"), "wb") as f: pickle.dump(malicious_users, f)

print(f"\n✅ Preprocessing v2 complete! Features: {len(feature_cols)}")
print("   Next step: caffeinate -i python3 lstm_training.py")