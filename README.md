
 🚀 Exoplanet Hunter — Godmode Edition

**Team:** the_elites  
**Challenge:** NASA Space Apps 2025 — *A World Away: Hunting for Exoplanets with AI*  
**Project Type:** AI-based Exoplanet Detection & Ensemble Prediction  
**Google Drive (Full Project Download):** [📂 Exoplanet Godmode](https://drive.google.com/drive/folders/1erCh8nY-Avo2AVxZ8ASwljVQyY2pk9Pm?usp=drive_link)

---

## 🌌 Overview

**Exoplanet Hunter (Godmode)** is a deep learning pipeline designed for NASA’s **“A World Away: Hunting for Exoplanets with AI”** challenge.

We engineered a multi-model ensemble that detects and classifies exoplanets using NASA’s open Kepler and TESS datasets.  
The system leverages multiple architectures — each specialized for temporal or spatial signal extraction — then fuses them using a **weighted ensemble inference** for precision and stability.

This setup was trained and benchmarked using **TensorDock cloud servers**, enabling ultra-fast large-scale training runs on heavy GPU compute.

---

## ⚙️ Hardware Setup (TensorDock Cloud)

To train and fine-tune our deep-learning architectures efficiently, we rented a **TensorDock H100-SXM compute instance** featuring:

| Component | Specification |
|------------|----------------|
| **GPU** | NVIDIA H100 SXM (80 GB VRAM) |
| **CPU** | 60-core virtualized high-frequency processor |
| **RAM** | 80 GB DDR5 |
| **Storage** | 1 TB NVMe SSD |
| **OS** | Ubuntu 22.04 LTS |
| **Frameworks** | PyTorch, Transformers, Scikit-learn |

This setup allowed us to parallelize model training, hyperparameter tuning (via Optuna), and ensemble fusion on large datasets (8GB+ processed).

---

## 🧠 Model Architectures Used

Our ensemble was designed with six complementary architectures, each targeting unique signal patterns found in exoplanet light curves.

| Model | Type | Purpose | Highlights |
|--------|------|----------|-------------|
| **CNN_1D** | Convolutional Neural Network | Detects local patterns and periodic dips in stellar brightness. | Fast, great for time-series with clear trends. |
| **BiLSTM_Attention** | Bidirectional LSTM + Attention Layer | Captures long-range dependencies in light curves. | Learns sequential dependencies & temporal context. |
| **GRU_Stacked** | Gated Recurrent Unit Network | Lightweight sequence model for faster training. | Excellent generalization and efficiency. |
| **TCN (Temporal Convolutional Network)** | Causal convolution-based temporal model | Learns multi-scale features from irregular sequences. | Robust for uneven sampling in astronomical data. |
| **Transformer_Encoder** | Self-attention transformer | Extracts relational patterns across the entire sequence. | Provides global context awareness. |
| **XGBoost Meta-Classifier** | Gradient-boosted ensemble | Combines outputs of all models to form the final prediction. | Acts as the “brain” of the ensemble. |

---

## ⚗️ How the Ensemble Works

1. **Training Stage:**
   - Each neural network model was trained independently on the same dataset split (`train.npz`, `val.npz`, `test.npz`).
   - Loss functions were tuned individually (MSE / CrossEntropy depending on architecture).
   - Best checkpoints were automatically saved into `/data` as `.pt` files.

2. **Inference Stage:**
   - All model predictions (`pred_0`, `pred_1`) were collected and averaged using a **weighted ensemble mean**.
   - Ensemble predictions were saved to:
     - `ensemble_predictions.npy`
     - `ensemble_predictions.csv`

3. **Conversion Stage:**
   - `Convert.py` translated binary model outputs into human-readable numeric predictions, stored as a `.csv` file for analysis and visualization.

This architecture ensures **robust detection**, reducing false positives and outperforming single-model baselines.

---

## 🧩 Directory Layout

exoplanet_godmode_final/
│
├── debug1.sh # Environment setup & preprocessing script
├── requirements.txt # Dependency list
│
├── py/
│ ├── train.py # Multi-model training
│ ├── inference.py # Ensemble inference
│ ├── models_full.py # All architectures defined
│ ├── datautils.py # Data handling utilities
│ └── ...
│
├── data/
│ ├── train.npz
│ ├── val.npz
│ ├── test.npz
│ ├── cnn_model.pt
│ ├── rnn_model.pt
│ ├── ensemble_predictions.npy
│ ├── ensemble_predictions.csv
│ └── ...
│
├── Convert.py # Converts .npy predictions to .csv
└── logs/
└── run.log


---

## 🧾 Requirements

All dependencies are listed in `requirements.txt`:

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


Install using:
```bash
pip install -r requirements.txt

🔧 Setup and Usage

    ⚠️ Run each command in order. Don’t skip steps unless noted.

1️⃣ Move to Your Desired Directory

cd ~/Desktop

If you downloaded the project from Google Drive:

# Extract it first
cd exoplanet_godmode_final

2️⃣ Install Dependencies

pip install -r requirements.txt

3️⃣ Setup Environment & Prepare Data

bash debug1.sh

This will set up the virtual environment, install all dependencies, and prepare the .npz training data.
4️⃣ Train Models (Optional)

If you want to retrain all models:

cd py
python3 train.py

Model weights (.pt files) will be saved automatically in /data.

If you already have trained models, skip to Step 5.
5️⃣ Run Inference (Generate Predictions)

python3 inference.py

Output:

/data/ensemble_predictions.npy
/data/ensemble_predictions.csv

6️⃣ Convert Predictions to CSV (Readable Format)

cd ..
cd data
python3 Convert.py

Expected output:

Shape: (566, 2)
First 10 predictions:
[[2140.0645 -1573.006]
 [183.9472  -137.2675]
  ... ]

7️⃣ View Final Predictions

cat ensemble_predictions.csv
# or open with any spreadsheet tool

🌍 NASA Challenge Context

    NASA Space Apps Challenge 2025
    Theme: A World Away — Hunting for Exoplanets with AI
    Official Challenge Page

Our mission:

    Use AI to identify exoplanets from light curves captured by NASA’s Kepler and TESS missions, developing a scalable open-source framework for future researchers and citizen scientists.

👥 Team: the_elites
Role	Description
Lead Developer	Built and optimized all deep learning models.
Data Engineer	Handled dataset preprocessing and splitting logic.
ML Ops Engineer	Managed TensorDock cloud infrastructure and deployment.
Analyst	Validated predictions and ensemble weighting.

We are the_elites, a team of passionate space-AI enthusiasts working to extend the frontier of exoplanet research through intelligent automation. 🌌
🧩 Summary — Full Command Flow

For convenience, here’s everything together 👇

cd ~/Desktop
# (Download project from Drive)
cd exoplanet_godmode_final
pip install -r requirements.txt
bash debug1.sh
cd py
python3 train.py      # optional
python3 inference.py
cd ..
cd data
python3 Convert.py
cat ensemble_predictions.csv

🌠 Conclusion

Exoplanet Hunter — Godmode Edition represents a complete, production-ready exoplanet detection pipeline.
It merges deep learning, data science, and astrophysics — trained on state-of-the-art cloud hardware — to detect worlds light-years away.

Developed by the_elites for NASA Space Apps Challenge 2025.

    "AI doesn’t just look at the stars — it helps us find new ones." ✨
