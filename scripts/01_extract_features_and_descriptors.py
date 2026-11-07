import os
import sys
import argparse
import subprocess
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from src.transferability_estimators import compute_all_estimators

DATASETS = {
    'Kermany':  '/kaggle/input/datasets/imran50/kermany-pcxr/chest_xray',
    'PRF':      '/kaggle/input/datasets/imran50/sylhet-train-test/sylhet/chest_xray',
    'NIH-RSNA': '/kaggle/input/datasets/imran50/nih-rsna-acxr',
    'CheXpert': '/kaggle/input/datasets/imran50/chexpert',
    'PadChest': '/kaggle/input/datasets/imran50/padchest-8bit',
}
SPLIT = 'train'
EXTRACTORS = [
    ('nih',  'densenet121-res224-nih'),
    ('chex', 'densenet121-res224-chex'),
]
IMG_SIZE = (224, 224)
BATCH_SIZE = 64
SEED = 42

def preprocess_image(img):
    arr = np.array(img.convert('L').resize(IMG_SIZE), dtype=np.float32)
    return ((arr / 255.0) * 2048.0 - 1024.0)[None, :, :]

def extract_folder(folder, embed_fn):
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg')))
    feats, batch, n_failed = [], [], 0
    for fname in tqdm(files, desc=os.path.basename(folder)):
        try:
            with Image.open(os.path.join(folder, fname)) as img:
                batch.append(preprocess_image(img))
        except Exception:
            n_failed += 1
            continue
        if len(batch) == BATCH_SIZE:
            feats.append(embed_fn(np.stack(batch)))
            batch = []
    if batch:
        feats.append(embed_fn(np.stack(batch)))
    return np.vstack(feats).astype(np.float32), n_failed, len(files)

def build_embedder(weights):
    try:
        import torchxrayvision as xrv
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'torchxrayvision'])
        import torchxrayvision as xrv
    import torch
    import torch.nn as nn
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = xrv.models.DenseNet(weights=weights)
    net = nn.Sequential(model.features, nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()).eval().to(dev)
    def embed(b):
        with torch.inference_mode():
            return net(torch.from_numpy(b).to(dev)).cpu().numpy()
    return embed

def main():
    os.makedirs('outputs', exist_ok=True)
    for tag, weights in EXTRACTORS:
        print(f'Starting extraction with {weights} ({tag})...')
        embed_fn = build_embedder(weights)
        records = []
        for name, root in DATASETS.items():
            n_dir = os.path.join(root, SPLIT, 'NORMAL')
            p_dir = os.path.join(root, SPLIT, 'PNEUMONIA')
            if not os.path.exists(n_dir):
                print(f'[SKIP] {n_dir} not found')
                continue
            X0, f0, len0 = extract_folder(n_dir, embed_fn)
            X1, f1, len1 = extract_folder(p_dir, embed_fn)
            X = np.vstack([X0, X1])
            y = np.concatenate([np.zeros(len(X0)), np.ones(len(X1))]).astype(int)
            res = compute_all_estimators(X, y, seed=SEED)
            res.update({
                'Dataset': name,
                'Extractor': tag,
                'Samples': len(X),
                'N_normal': len(X0),
                'N_pneumonia': len(X1),
                'N_failed_normal': f0,
                'N_failed_pneumonia': f1,
                'Feature_Dim': X.shape[1],
            })
            records.append(res)
        df = pd.DataFrame(records)
        out_path = f'outputs/feature_space_analysis_{tag}.csv'
        df.to_csv(out_path, index=False)
        print(f'Saved: {out_path}')

if __name__ == '__main__':
    main()
