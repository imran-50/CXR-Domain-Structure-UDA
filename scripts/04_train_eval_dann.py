import os
import json
import copy
import time
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.models as models
from torchvision import transforms
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from configs.dataset_config import DATASETS, DATASET_NAMES
from src.models_dann import DANN

SEED = 42
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CONFIG = {
    "image_size": 224, "batch_size": 32, "epochs": 30, "lr": 1e-4,
    "early_stop_patience": 7, "grad_clip_max_norm": 5.0,
    "scheduler_patience": 3, "scheduler_factor": 0.5, "weight_decay": 1e-5
}

ALL_PAIRS = [(s, t) for s in DATASET_NAMES for t in DATASET_NAMES if s != t]
OUTPUT_ROOT = "outputs/dann_runs"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

class ChestXRayDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {"NORMAL": 0, "PNEUMONIA": 1}

        if os.path.exists(root_dir):
            for class_name in ["NORMAL", "PNEUMONIA"]:
                cdir = os.path.join(root_dir, class_name)
                if os.path.exists(cdir):
                    for fname in os.listdir(cdir):
                        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.samples.append((os.path.join(cdir, fname), self.class_to_idx[class_name]))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform: image = self.transform(image)
        return image, label

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

train_tx = transforms.Compose([
    transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])
eval_tx = transforms.Compose([
    transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

def evaluate_loader(model, loader):
    model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        for x, y in loader:
            class_preds, _ = model(x.to(device))
            probs = F.softmax(class_preds, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(y.numpy())
    try: auc = roc_auc_score(all_targets, all_probs)
    except ValueError: auc = float('nan')
    return auc

def run_pair(s_name, t_name):
    run_dir = os.path.join(OUTPUT_ROOT, f"{s_name}__to__{t_name}")
    if os.path.exists(os.path.join(run_dir, "metrics.json")):
        return json.load(open(os.path.join(run_dir, "metrics.json")))

    os.makedirs(run_dir, exist_ok=True)
    src_train = ChestXRayDataset(os.path.join(DATASETS[s_name], "train"), transform=train_tx)
    tgt_train = ChestXRayDataset(os.path.join(DATASETS[t_name], "train"), transform=eval_tx)
    tgt_test  = ChestXRayDataset(os.path.join(DATASETS[t_name], "test"), transform=eval_tx)

    # Class balanced sampler
    labels = [l for _, l in src_train.samples]
    counts = np.bincount(labels, minlength=2)
    weights = [1.0 / max(counts[l], 1) for l in labels]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    src_loader = DataLoader(src_train, batch_size=CONFIG["batch_size"], sampler=sampler, drop_last=True)
    tgt_loader = DataLoader(tgt_train, batch_size=CONFIG["batch_size"], shuffle=True, drop_last=True)
    test_loader = DataLoader(tgt_test, batch_size=CONFIG["batch_size"], shuffle=False)

    model = DANN(num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    best_auc, patience = -1.0, 0
    print(f"Starting DANN Adaptation: {s_name} -> {t_name}")

    for epoch in range(CONFIG["epochs"]):
        model.train()
        len_src = len(src_loader)
        tgt_iter = iter(tgt_loader)

        for b_idx, (s_imgs, s_lbls) in enumerate(src_loader):
            p = float(b_idx + epoch * len_src) / (CONFIG["epochs"] * len_src)
            alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0

            try: t_imgs, _ = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader); t_imgs, _ = next(tgt_iter)

            s_imgs, s_lbls = s_imgs.to(device), s_lbls.to(device)
            t_imgs = t_imgs.to(device)

            optimizer.zero_grad()
            c_preds, d_preds_s = model(s_imgs, alpha=alpha)
            _, d_preds_t       = model(t_imgs, alpha=alpha)

            loss_c = criterion(c_preds, s_lbls)
            loss_d = criterion(d_preds_s, torch.zeros(s_imgs.size(0), dtype=torch.long, device=device)) + \
                     criterion(d_preds_t, torch.ones(t_imgs.size(0), dtype=torch.long, device=device))

            (loss_c + loss_d).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip_max_norm"])
            optimizer.step()

        auc = evaluate_loader(model, test_loader)
        if auc > best_auc:
            best_auc, patience = auc, 0
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pth"))
        else:
            patience += 1
            if patience >= CONFIG["early_stop_patience"]:
                break

    record = {"Source": s_name, "Target": t_name, "AUC": float(best_auc)}
    json.dump(record, open(os.path.join(run_dir, "metrics.json"), "w"), indent=2)
    return record

def main():
    results = []
    for s, t in ALL_PAIRS:
        try:
            r = run_pair(s, t)
            results.append(r)
            print(f"[DONE] {s} -> {t}: AUC = {r['AUC']:.4f}")
        except Exception as e:
            print(f"[ERR] {s} -> {t}: {e}")

    pd.DataFrame(results).to_csv("outputs/dann_summary_all_pairs.csv", index=False)

if __name__ == "__main__":
    main()
