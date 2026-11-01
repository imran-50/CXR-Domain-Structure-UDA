import os
import sys
import json
import argparse
import subprocess
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from configs.dataset_config import DATASETS

SPLIT = "train"
EXTRACTORS = [
    ("chex", "densenet121-res224-chex"),
    ("nih",  "densenet121-res224-nih"),
]
PRIMARY_TAG = EXTRACTORS[0][0]
IMG_SIZE = (224, 224)
BATCH_SIZE = 64
RANDOM_SEED = 42
CACHE_DIR = "cached_features"
OUT_DIR = "outputs"
GBC_VAR_FLOOR_REL = 1e-6

np.random.seed(RANDOM_SEED)

def preprocess_image(img):
    arr = np.array(img.convert("L").resize(IMG_SIZE), dtype=np.float32)
    return ((arr / 255.0) * 2048.0 - 1024.0)[None, :, :]

def extract_folder_features(folder, embed_fn, batch_size=BATCH_SIZE):
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    feats, batch, n_failed = [], [], 0

    def flush():
        nonlocal batch
        if batch:
            feats.append(np.asarray(embed_fn(np.stack(batch).astype(np.float32))))
            batch = []

    for fname in tqdm(files, desc=os.path.basename(folder)):
        try:
            with Image.open(os.path.join(folder, fname)) as img:
                batch.append(preprocess_image(img))
        except Exception:
            n_failed += 1
            continue
        if len(batch) == batch_size:
            flush()
    flush()
    return np.vstack(feats).astype(np.float32), n_failed, len(files)

def build_embedder(weights):
    try:
        import torchxrayvision as xrv
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torchxrayvision"])
        import torchxrayvision as xrv
    import torch
    import torch.nn as nn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = xrv.models.DenseNet(weights=weights)
    extractor = nn.Sequential(
        model.features, nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
    ).eval().to(device)

    def embed(batch_np):
        with torch.inference_mode():
            return extractor(torch.from_numpy(batch_np).to(device)).cpu().numpy()

    return embed

def gbc_pca_spherical(X, y, n_components=64):
    d = int(min(n_components, X.shape[0] - 1, X.shape[1]))
    Z = PCA(n_components=d, random_state=RANDOM_SEED).fit_transform(X)
    Z0, Z1 = Z[y == 0], Z[y == 1]
    mu0, mu1 = Z0.mean(axis=0), Z1.mean(axis=0)
    s0, s1 = Z0.var(axis=0, ddof=1).mean(), Z1.var(axis=0, ddof=1).mean()
    m = 0.5 * (s0 + s1)
    db = float(np.sum((mu0 - mu1) ** 2) / (8.0 * m) + 0.5 * d * np.log(m / np.sqrt(s0 * s1)))
    return 2.0 * db

def evaluate_dataset(name, normal_features, pneumonia_features, tag):
    X_raw = np.vstack([normal_features, pneumonia_features]).astype(np.float64)
    y = np.concatenate([np.zeros(len(normal_features)), np.ones(len(pneumonia_features))]).astype(int)
    X_scaled = StandardScaler().fit_transform(X_raw)
    n0 = len(normal_features)
    X0s, X1s = X_scaled[:n0], X_scaled[n0:]

    sil = float(silhouette_score(X_scaled, y, sample_size=min(3000, len(X_scaled)), random_state=42))
    db = float(davies_bouldin_score(X_scaled, y))
    between = np.sum((X0s.mean(axis=0) - X1s.mean(axis=0)) ** 2)
    within = np.sum(np.var(X0s, axis=0, ddof=1)) + np.sum(np.var(X1s, axis=0, ddof=1))
    fisher = float(between / (within + 1e-12))
    gbc = gbc_pca_spherical(X_raw, y)

    return {
        "Dataset": name,
        "Silhouette": sil,
        "Davies_Bouldin": db,
        "Fisher_Ratio": fisher,
        "GBC": gbc,
        "N_normal": len(normal_features),
        "N_pneumonia": len(pneumonia_features),
    }

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for tag, weights in EXTRACTORS:
        embedder = build_embedder(weights)
        rows = []
        for name, root in DATASETS.items():
            norm_dir = os.path.join(root, SPLIT, "NORMAL")
            pneu_dir = os.path.join(root, SPLIT, "PNEUMONIA")
            if not os.path.exists(norm_dir):
                print(f"[SKIP] Directory not found for {name}: {norm_dir}")
                continue
            nf, _, _ = extract_folder_features(norm_dir, embedder)
            pf, _, _ = extract_folder_features(pneu_dir, embedder)
            rows.append(evaluate_dataset(name, nf, pf, tag))

        if rows:
            df = pd.DataFrame(rows)
            z = lambda v: (v - v.mean()) / v.std(ddof=1)
            df["Cluster_Score"] = z(df["Silhouette"]) + z(df["Fisher_Ratio"]) - z(df["Davies_Bouldin"])
            df.to_csv(os.path.join(OUT_DIR, f"feature_space_analysis_{tag}.csv"), index=False)
            print(f"Saved: {OUT_DIR}/feature_space_analysis_{tag}.csv")

if __name__ == "__main__":
    main()
