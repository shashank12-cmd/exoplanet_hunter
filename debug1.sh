#!/usr/bin/env bash
# ================= Debug Installer & NASA Dataset Downloader =================
# Safe for 8GB RAM servers; step-by-step checks; fully automated for multiple datasets

set -euo pipefail
IFS=$'\n\t'

# ---------------- Paths ----------------
ROOT="$HOME/Desktop/exoplanet_godmode_final"
PYDIR="$ROOT/py"
DATADIR="$ROOT/data"
OUTDIR="$ROOT/outputs"
LOGDIR="$ROOT/logs"
VENV="$ROOT/venv"
MODELS_DIR="$OUTDIR/models"
PLOTS_DIR="$OUTDIR/plots"
KAGGLE_LOCAL="$HOME/Desktop/kaggle.json"

mkdir -p "$PYDIR" "$DATADIR" "$OUTDIR" "$LOGDIR" "$MODELS_DIR" "$PLOTS_DIR"
LOGFILE="$LOGDIR/run.log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "==== Debug Installer starting ===="
echo "Project root: $ROOT"
date

# ---------------- Python setup ----------------
PYTHON_BIN="$(command -v python3.10 || command -v python3 || true)"
if [ -z "$PYTHON_BIN" ]; then
    echo "ERROR: python3 not found. Install python3.10 or python3."
    exit 1
fi
echo "Using Python: $($PYTHON_BIN --version 2>&1)"

# ---------------- Create virtual environment ----------------
if [ ! -d "$VENV" ]; then
    echo "Creating venv at $VENV"
    $PYTHON_BIN -m venv "$VENV"
fi
source "$VENV/bin/activate"
pip install --upgrade pip setuptools wheel

# ---------------- Core packages ----------------
echo "Installing core Python packages..."
pip install --no-cache-dir numpy pandas scikit-learn xgboost matplotlib seaborn tqdm joblib astropy lightkurve kaggle
pip install --no-cache-dir transformers accelerate peft bitsandbytes gradio requests openai boto3 optuna || true

# ---------------- Kaggle key ----------------
if [ ! -f "$KAGGLE_LOCAL" ]; then
    echo "Kaggle key not found at $KAGGLE_LOCAL. Place your kaggle.json here and rerun."
    exit 1
fi
chmod 600 "$KAGGLE_LOCAL"
export KAGGLE_CONFIG_DIR="$(dirname "$KAGGLE_LOCAL")"
echo "Kaggle key ready. Config dir exported: $KAGGLE_CONFIG_DIR"

# ---------------- Python modules ----------------
echo "Setting PYTHONPATH: $PYDIR"
export PYTHONPATH="$PYDIR:${PYTHONPATH:-}"

# ---------------- Data download function ----------------
download_nasa_dataset() {
    local dataset_ref=$1
    echo "Downloading $dataset_ref..."
    mkdir -p "$DATADIR"
    python3 - <<PY
import subprocess, os
DATA_DIR = os.path.join("$DATADIR")
os.makedirs(DATA_DIR, exist_ok=True)
try:
    subprocess.run([
        "kaggle", "datasets", "download", "-d", "$dataset_ref", "-p", DATA_DIR, "--unzip"
    ], check=True)
    print("✅ Dataset $dataset_ref downloaded and unzipped to", DATA_DIR)
except subprocess.CalledProcessError as e:
    print("❌ Kaggle download failed:", e)
PY
}

# ---------------- Download multiple NASA exoplanet datasets ----------------
NASA_DATASETS=(
    "keplersmachines/kepler-labelled-time-series-data"
    "nasa/kepler-exoplanet-search-results"
    "arashnic/exoplanets"
    "adityamishraml/nasaexoplanets"
    "shivamb/all-exoplanets-dataset"
)

for ds in "${NASA_DATASETS[@]}"; do
    download_nasa_dataset "$ds"
done


#---------CSV -> NPZ conversion ----------------
echo "Converting CSVs to NPZ..."
python3 - <<PY
import os, pandas as pd, numpy as np

DATA_DIR = "$DATADIR"
npz_file = os.path.join(DATA_DIR,'exo_train.npz')

csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
all_X, all_y = [], []
shapes = {}

for csv_file in csv_files:
    df = pd.read_csv(os.path.join(DATA_DIR,csv_file))
    if 'LABEL' not in df.columns:
        df['LABEL'] = df.iloc[:,-1]

    # Select only numeric columns as features
    X = df.select_dtypes(include=['float64','int64']).values.astype('float32')

    # Skip empty or tiny datasets
    if X.shape[1] < 10:
        print(f"⚠️ Skipping {csv_file}: only {X.shape[1]} numeric cols.")
        continue

    # Map text labels to integers
    labels, y = np.unique(df['LABEL'], return_inverse=True)
    y = y.astype('int64')

    shapes[X.shape[1]] = shapes.get(X.shape[1], 0) + 1
    all_X.append(X)
    all_y.append(y)

# --- Detect dominant feature size ---
if not all_X:
    raise RuntimeError("❌ No valid CSVs found with numeric data.")

common_dim = max(shapes, key=shapes.get)
print(f"🧠 Most common feature length: {common_dim}")

# --- Filter datasets that match common_dim ---
filtered_X, filtered_y = [], []
for X, y in zip(all_X, all_y):
    if X.shape[1] == common_dim:
        filtered_X.append(X)
        filtered_y.append(y)
    else:
        print(f"⚠️ Skipping dataset with shape {X.shape}")

# --- Concatenate safely ---
X_all = np.concatenate(filtered_X, axis=0)
y_all = np.concatenate(filtered_y, axis=0)

np.savez_compressed(npz_file, X=X_all, y=y_all)
print(f"✅ Combined CSVs saved to {npz_file} | Final shape: {X_all.shape}")
PY



# ---------------- Create train/val/test splits ----------------
echo "Creating train/val/test splits..."
python3 - <<PY
import numpy as np
from sklearn.model_selection import train_test_split
import os

DATA_DIR = "$DATADIR"
d = np.load(os.path.join(DATA_DIR,'exo_train.npz'))
X, y = d['X'], d['y']

# Stratified split
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1111, stratify=y_temp, random_state=42)  # 10% val

np.savez_compressed(os.path.join(DATA_DIR,'train.npz'), X=X_train, y=y_train)
np.savez_compressed(os.path.join(DATA_DIR,'val.npz'), X=X_val, y=y_val)
np.savez_compressed(os.path.join(DATA_DIR,'test.npz'), X=X_test, y=y_test)
print(f"✅ Splits created | Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
PY

echo "==== Debug1.sh completed. Ready to run training & inference! ===="
