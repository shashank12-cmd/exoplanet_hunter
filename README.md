# 🚀 Exoplanet Hunter — Godmode Edition

**Team:** the_elites  
**Challenge:** NASA Space Apps 2025 — *A World Away: Hunting for Exoplanets with AI*  
**Project Type:** AI-based Exoplanet Detection & Ensemble Prediction  
**Google Drive (Full Project Download):** [📂 Exoplanet Godmode](https://drive.google.com/drive/folders/1erCh8nY-Avo2AVxZ8ASwljVQyY2pk9Pm?usp=drive_link)

---

## 🌌 Overview

This project — **Exoplanet Hunter (Godmode)** — is a full deep-learning framework designed for the NASA Space Apps Challenge **“A World Away: Hunting for Exoplanets with AI.”**  
We developed an ensemble pipeline using **CNNs, GRUs, Transformers, TCNs, LSTM-Attention**, and other hybrid architectures to classify potential exoplanets from NASA’s public datasets.

Everything from **data preprocessing → model training → inference → CSV export** is automated and reproducible.

---

## 🧠 Core Components

| Component | Description |
|------------|-------------|
| **debug1.sh** | Sets up the full environment (installs dependencies, prepares data, and generates train/val/test splits). |
| **train.py** | Trains multiple deep-learning models and saves their weights into the `/data` folder. |
| **inference.py** | Loads trained models, performs ensemble inference, and saves predictions as `.npy` and `.csv`. |
| **Convert.py** | Converts ensemble predictions into readable `.csv` and prints the summary to terminal. |

---

## ⚙️ Installation & Setup (Step-by-Step)

> 💡 Follow these commands **exactly** in your terminal. Each step is important — no shortcuts.

---

### 🪐 1. Clone or Download Project

If you’re using Google Drive (recommended because GitHub can’t handle the large files):

```bash
# Go to your Desktop or any location you prefer
cd ~/Desktop

# Download from Google Drive link (manually or using your browser)
# https://drive.google.com/drive/folders/1erCh8nY-Avo2AVxZ8ASwljVQyY2pk9Pm?usp=drive_link

# After extracting or syncing it, navigate into the project folder
cd exoplanet_godmode_final

If cloning from GitHub (optional):

git clone <your-github-repo-url>
cd exoplanethunter

🧩 2. Install All Requirements

pip install -r requirements.txt

    ⚠️ Use Python 3.10+ for full compatibility (recommended: Python 3.10.12)

🛰️ 3. Run the Debug Installer Script

This script will:

    Verify your Python installation

    Set up a virtual environment if needed

    Install all core dependencies

    Prepare /data folder with .npz splits (train, val, test)

Run it like this:

bash debug1.sh

🧬 4. Train the Models (optional)

If you want to retrain all models yourself:

cd py
python3 train.py

Model checkpoints (*.pt files) will be saved in:

/home/<user>/Desktop/exoplanet_godmode_final/data/

If you already have trained models, you can skip this step.
🌠 5. Run Inference

Whether you trained your models or already have them:

python3 inference.py

This will:

    Load all .pt model weights from /data

    Run ensemble predictions

    Save outputs to:

        /data/ensemble_predictions.npy

        /data/ensemble_predictions.csv

🪄 6. Convert Predictions to CSV (Readable Format)

Now move to your data directory:

cd ..
cd data

Run the converter script:

python3 Convert.py

You’ll see:

Shape: (566, 2)
First 10 predictions:
[[ 2140.0645  -1573.006  ]
 [  183.9472   -137.2675 ]
  ... ]

This also creates a file called:

ensemble_predictions.csv

You can open it using:

libreoffice ensemble_predictions.csv
# or
cat ensemble_predictions.csv

🗂 Directory Structure

exoplanet_godmode_final/
│
├── debug1.sh
├── requirements.txt
│
├── py/
│   ├── train.py
│   ├── inference.py
│   ├── models_full.py
│   ├── datautils.py
│   └── ...
│
├── data/
│   ├── train.npz
│   ├── val.npz
│   ├── test.npz
│   ├── cnn_model.pt
│   ├── rnn_model.pt
│   ├── ...
│   ├── ensemble_predictions.npy
│   └── ensemble_predictions.csv
│
├── Convert.py
└── logs/
    └── run.log

🧾 Requirements Summary

Here’s what’s included in requirements.txt:

numpy
pandas
matplotlib
scikit-learn
xgboost
tqdm
joblib
torch
torchvision
astropy
lightkurve
transformers
accelerate
peft
bitsandbytes
gradio
optuna
requests
boto3

🌍 Challenge Context

    NASA Space Apps Challenge 2025
    Theme: A World Away — Hunting for Exoplanets with AI
    Official Challenge Page

Our objective:

    Build a robust AI framework capable of detecting exoplanets using public NASA datasets and deep learning architectures — exploring how AI can accelerate planetary discovery.

👥 Team — the_elites

We are the_elites, a small but passionate team dedicated to pushing the limits of AI in space science.
Our focus is to blend innovation, simplicity, and accuracy — empowering future astronomers and AI researchers to discover new worlds. 🌌
✅ Quick Recap (All Commands Together)

For reference, here’s the full one-shot flow 👇

# Step 1 — Move to Desktop or target location
cd ~/Desktop

# Step 2 — Download the full project (Google Drive)
# https://drive.google.com/drive/folders/1erCh8nY-Avo2AVxZ8ASwljVQyY2pk9Pm?usp=drive_link

# Step 3 — Enter the project directory
cd exoplanet_godmode_final

# Step 4 — Install dependencies
pip install -r requirements.txt

# Step 5 — Run setup script
bash debug1.sh

# Step 6 — (Optional) Train models
cd py
python3 train.py

# Step 7 — Run inference
python3 inference.py

# Step 8 — Convert predictions to CSV
cd ..
cd data
python3 Convert.py

# Step 9 — View final predictions
cat ensemble_predictions.csv

🌟 Mission Complete

The Exoplanet Hunter (Godmode) system is now fully operational.
You can train, infer, or directly explore predictions — all without external dependencies.

Built by the_elites — for NASA Space Apps 2025.
Exploring worlds beyond our own. 🌌
