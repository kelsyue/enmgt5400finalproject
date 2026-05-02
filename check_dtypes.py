import pandas as pd
import os

DATA_DIR = "r6.2/"
activity_files = ["logon.csv", "device.csv", "email.csv", "file.csv", "http.csv"]

for f in activity_files:
    path = os.path.join(DATA_DIR, f)
    # Read a few rows with parse_dates
    df = pd.read_csv(path, nrows=100, parse_dates=["date"])
    print(f"{f} date dtype: {df['date'].dtype}")
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        print(f"  Sample values: {df['date'].head().tolist()}")
