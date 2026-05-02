"""
CERT r6.2 Insider Threat Dataset — Exploratory Data Analysis (Week 1)

This script performs exploratory analysis of the CERT r6.2 dataset,
examining event frequencies, class imbalance, and malicious vs. benign
activity patterns across five log types.

Activity Types Analyzed:
    - logon:   authentication events
    - device:  USB device usage (potential exfiltration vector)
    - email:   internal/external email traffic
    - file:    file operations and copies
    - http:    web traffic

Outputs:
    - event_frequency.png: daily event counts by activity type
    - class_imbalance.png: benign vs malicious event distribution
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os

# --- Config ---
DATA_DIR    = "r6.2/"
ANSWERS_DIR = "answers/"

sns.set_theme(style="darkgrid")
plt.rcParams["figure.figsize"] = (14, 5)

print("=" * 60)
print(" CERT r6.2 — Week 1: Data Analysis")
print("=" * 60)

# Ground Truth Labels
# Load insider threat ground-truth: each row is a confirmed malicious user
# with labeled start/end dates of their exfiltration campaign.
print("\n[1/6] Loading ground truth labels...")
insiders = pd.read_csv(os.path.join(ANSWERS_DIR, "insiders.csv"))

# Keep only CERT r6.2 dataset entries
r62 = insiders[insiders["dataset"] == 6.2].copy()
r62["start"] = pd.to_datetime(r62["start"], format="mixed", errors="coerce")
r62["end"]   = pd.to_datetime(r62["end"],   format="mixed", errors="coerce")

malicious_users = set(r62["user"].unique())
print(f"  r6.2 malicious users ({len(malicious_users)}): {sorted(malicious_users)}")
print(r62[["user", "scenario", "start", "end"]].to_string(index=False))

# Load Activity CSVs
# Read raw logs for each activity type; date parsing enables time-based analysis.
print("\n[2/6] Loading activity CSVs (this may take a few minutes)...")
logon   = pd.read_csv(os.path.join(DATA_DIR, "logon.csv"),  parse_dates=["date"])
device  = pd.read_csv(os.path.join(DATA_DIR, "device.csv"), parse_dates=["date"])
email   = pd.read_csv(os.path.join(DATA_DIR, "email.csv"),  parse_dates=["date"])
file_df = pd.read_csv(os.path.join(DATA_DIR, "file.csv"),   parse_dates=["date"])
http    = pd.read_csv(os.path.join(DATA_DIR, "http.csv"),   parse_dates=["date"])

activity_dfs = {
    "logon": logon, "device": device, "email": email,
    "file": file_df, "http": http,
}

total = sum(len(df) for df in activity_dfs.values())
for name, df in activity_dfs.items():
    print(f"  {name:<8} {len(df):>12,} rows, date dtype: {df['date'].dtype}")
print(f"  {'TOTAL':<8} {total:>12,} rows")

# User Overview
# Build sets of all users and separate benign vs. malicious users.
print("\n[3/6] User overview...")
all_users    = set().union(*[set(df["user"]) for df in activity_dfs.values()])
benign_users = all_users - malicious_users
print(f"  Total users:     {len(all_users)}")
print(f"  Malicious users: {len(malicious_users)}")
print(f"  Benign users:    {len(benign_users)}")
print(f"  Malicious ratio: {len(malicious_users) / len(all_users) * 100:.2f}%")

# Event Frequency Over Time
# Plot daily event counts for each activity type to reveal temporal patterns
# and potential periodicity or spikes related to malicious activity.
print("\n[4/6] Plotting event frequency over time...")
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
fig, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=True)

for ax, (name, df), color in zip(axes, activity_dfs.items(), colors):
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        print(f"  Converting {name} date column to datetime...")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    
    daily = df.groupby(df["date"].dt.date).size()
    ax.plot(pd.to_datetime(daily.index), daily.values, color=color, linewidth=0.8)
    ax.set_ylabel("Events", fontsize=10)
    ax.set_title(f"{name} — daily event count", fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

plt.suptitle("CERT r6.2 — Daily Event Frequency by Activity Type",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("event_frequency.png", dpi=150, bbox_inches="tight")
print("  Saved event_frequency.png")

# Class Imbalance
# Quantify and visualize the severe class imbalance (benign vs malicious events)
# to highlight the challenge for anomaly detection models.
print("\n[5/6] Class imbalance analysis...")
rows = []
for name, df in activity_dfs.items():
    total_e     = len(df)
    malicious_e = df["user"].isin(malicious_users).sum()
    rows.append({
        "Activity":         name,
        "Total Events":     total_e,
        "Malicious Events": malicious_e,
        "Benign Events":    total_e - malicious_e,
        "Malicious %":      round(malicious_e / total_e * 100, 4),
    })

imbalance_df = pd.DataFrame(rows)
totals = imbalance_df[["Total Events", "Malicious Events", "Benign Events"]].sum()
totals["Activity"]    = "TOTAL"
totals["Malicious %"] = round(totals["Malicious Events"] / totals["Total Events"] * 100, 4)
imbalance_df = pd.concat([imbalance_df, pd.DataFrame([totals])], ignore_index=True)
print(imbalance_df.to_string(index=False))

plot_df = imbalance_df[imbalance_df["Activity"] != "TOTAL"].copy()
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

x = np.arange(len(plot_df))
axes[0].bar(x, plot_df["Benign Events"],    label="Benign",    color="#4C72B0")
axes[0].bar(x, plot_df["Malicious Events"], label="Malicious", color="#C44E52",
            bottom=plot_df["Benign Events"])
axes[0].set_xticks(x)
axes[0].set_xticklabels(plot_df["Activity"])
axes[0].set_title("Event Count by Activity Type")
axes[0].set_ylabel("Events")
axes[0].legend()

axes[1].bar(plot_df["Activity"], plot_df["Malicious %"], color="#C44E52")
axes[1].set_title("Malicious Event % by Activity Type")
axes[1].set_ylabel("Malicious %")
for i, v in enumerate(plot_df["Malicious %"]):
    axes[1].text(i, v + 0.01, f"{v:.3f}%", ha="center", fontsize=9)

plt.suptitle("Class Imbalance — CERT r6.2", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("class_imbalance.png", dpi=150, bbox_inches="tight")
print("  Saved class_imbalance.png")

# Malicious User Deep Dive
# Compare each malicious user's activity volume against the benign-user average
# to surface outliers and relative exfiltration intensity per user.
print("\n[6/6] Malicious user activity vs benign average...")
for name, df in activity_dfs.items():
    user_counts = df.groupby("user").size()
    benign_mean = user_counts[~user_counts.index.isin(malicious_users)].mean()
    print(f"\n  {name.upper()} — avg benign user: {benign_mean:.1f} events")
    for u in sorted(malicious_users):
        count = user_counts.get(u, 0)
        print(f"    {u}: {count:>6} events  ({count / benign_mean:.1f}x avg)")