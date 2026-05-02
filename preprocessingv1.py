"""
CERT r6.2 Insider Threat Dataset — Preprocessing Pipeline (v1, Baseline)

This script implements the baseline preprocessing pipeline for the CERT r6.2
insider threat dataset. It extracts simple daily aggregate features from five
activity log types and prepares 30-day sliding-window sequences for LSTM
autoencoder training.

This v1 pipeline uses a simpler feature set compared to v2:
  - No per-user normalization (uses global StandardScaler instead)
  - Fewer derived features (no hour statistics, no after-hours breakdowns
    for device/file, no email attachments)
  - Suitable as a baseline to quantify improvements from richer features
    and per-user normalization in v2.

Feature Set (7 daily aggregates per user):
    - logon_count:          number of logon events
    - logon_after_hours:    logons outside 08:00-18:00
    - device_count:         USB device usage count
    - email_count:          number of emails sent/received
    - email_size_total:     total email size (bytes)
    - file_count:           file copy operations
    - http_count:           web requests

Processing Steps:
    1. Load ground-truth labels (malicious user incident windows)
    2. Extract per-user daily aggregates from each activity log (chunked)
    3. Build complete user-day grid and fill missing entries with 0
    4. Label days falling within malicious incident windows
    5. Split users: 70% train, 15% val, 15% test (benign-only for train/val)
    6. Fit StandardScaler on training features, transform all sets
    7. Build 30-day sliding-window sequences (X) and next-day labels (y)

Outputs (saved to processed/):
    - X_train.npy, X_val.npy, X_test.npy: (n_samples, 30, 7)
    - y_test.npy: binary labels for test sequences
    - u_test.npy: usernames for test sequences
    - scaler.pkl: fitted StandardScaler object
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
print(" CERT r6.2 — Week 2: Preprocessing v1 (Baseline)")
print("=" * 60)

# Load Labels
# Load ground truth: confirmed malicious users and their incident windows.
print("\n[1/5] Loading ground truth...")
insiders   = pd.read_csv(os.path.join(ANSWERS_DIR, "insiders.csv"))
r62_labels = insiders[insiders["dataset"] == 6.2].copy()
r62_labels["start"] = pd.to_datetime(r62_labels["start"], format="mixed", errors="coerce")
r62_labels["end"]   = pd.to_datetime(r62_labels["end"],   format="mixed", errors="coerce")
malicious_users = set(r62_labels["user"].unique())
print(f"  Malicious users ({len(malicious_users)}): {sorted(malicious_users)}")

# Build Daily Feature Vectors (Chunked)
# Extract simple daily aggregates from each activity log, processing in chunks
# to manage memory usage for large CSV files.
print("\n[2/5] Building daily feature vectors (chunked)...")

def process_csv_in_chunks(filename, agg_funcs, date_col="date"):
    path = os.path.join(DATA_DIR, filename)
    chunks = []
    for chunk in pd.read_csv(path, chunksize=10**6, parse_dates=[date_col]):
        chunk["day"] = chunk[date_col].dt.normalize()
        chunk_agg = chunk.groupby(["user", "day"]).agg(**agg_funcs).reset_index()
        chunks.append(chunk_agg)
    full_agg = pd.concat(chunks)
    return full_agg.groupby(["user", "day"]).sum().reset_index()

# Logon: count + after hours only (no hour mean/std)
print("  Processing logon.csv...")
logon_feat = process_csv_in_chunks("logon.csv", {
    "logon_count":       ("activity", "count"),
    "logon_after_hours": ("date", lambda x: ((x.dt.hour < 8) | (x.dt.hour >= 18)).sum()),
})

# Device: count only (no after_hours breakdown)
print("  Processing device.csv...")
device_feat = process_csv_in_chunks("device.csv", {
    "device_count": ("activity", "count"),
})

# Email: count + size only (no attachments)
print("  Processing email.csv...")
email_feat = process_csv_in_chunks("email.csv", {
    "email_count":      ("id",   "count"),
    "email_size_total": ("size", "sum"),
})

# File: count only (no after_hours)
print("  Processing file.csv...")
file_feat = process_csv_in_chunks("file.csv", {
    "file_count": ("id", "count"),
})

# HTTP: count only
print("  Processing http.csv (this may take a while)...")
http_feat = process_csv_in_chunks("http.csv", {
    "http_count": ("id", "count"),
})

# Merge into User-Day Grid
# Combine all activity features into a complete user-day matrix and fill missing entries with 0.
print("\n[3/5] Merging features into user-day grid...")
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

# Add Precise Labels
# Mark days that fall within any malicious incident window.
print("\n[4/5] Adding labels based on incident windows...")
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

# Split, Scale & Build Sequences
# Split users (not individual rows) into train/val/test sets, fit a global
# StandardScaler on training features, and build 30-day sliding-window sequences.
# Note: train/val contain only benign days; test contains all days.
print("\n[5/5] Splitting, scaling, and building sequences...")

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

# V1: global StandardScaler (fit on train population, applied to all)
scaler = StandardScaler()
train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
val_df[feature_cols]   = scaler.transform(val_df[feature_cols])
test_df[feature_cols]  = scaler.transform(test_df[feature_cols])

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
with open(os.path.join(OUT_DIR, "scaler.pkl"),          "wb") as f: pickle.dump(scaler,          f)
with open(os.path.join(OUT_DIR, "feature_cols.pkl"),    "wb") as f: pickle.dump(feature_cols,    f)
with open(os.path.join(OUT_DIR, "malicious_users.pkl"), "wb") as f: pickle.dump(malicious_users, f)

print(f"\nPreprocessing v1 complete! Features: {len(feature_cols)}")