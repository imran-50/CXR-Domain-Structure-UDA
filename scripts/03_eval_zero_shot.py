import os
import argparse
import pandas as pd
import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

from configs.dataset_config import DATASETS, DATASET_NAMES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

eval_transforms = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

class PneumoniaDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.class_to_idx = {'NORMAL': 0, 'PNEUMONIA': 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.loc[idx, 'image_path']
        label = self.class_to_idx[self.df.loc[idx, 'label']]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

def load_model(checkpoint_path):
    model = timm.create_model('convnextv2_tiny', pretrained=False, num_classes=2, in_chans=3)
    n_features = model.head.fc.in_features
    model.head.fc = nn.Sequential(
        nn.Linear(n_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 2)
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device).eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained source model .pth")
    parser.add_argument("--source", type=str, required=True, help="Name of trained source dataset")
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    records = []

    print(f"=== Zero-Shot Cross-Domain Evaluation (Source: {args.source}) ===")
    for target in DATASET_NAMES:
        test_dir = os.path.join(DATASETS[target], "test")
        if not os.path.exists(test_dir):
            print(f"[SKIP] Test path not found: {test_dir}")
            continue

        paths = glob(os.path.join(test_dir, "*/*.jpeg")) + glob(os.path.join(test_dir, "*/*.jpg")) + glob(os.path.join(test_dir, "*/*.png"))
        df = pd.DataFrame({'image_path': paths, 'label': [os.path.basename(os.path.dirname(p)) for p in paths]})
        loader = DataLoader(PneumoniaDataset(df, eval_transforms), batch_size=64, shuffle=False)

        all_preds, all_probs, all_labels = [], [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                logits = model(x)
                probs = torch.softmax(logits, dim=1)[:, 1]
                preds = torch.argmax(logits, dim=1)
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.numpy())

        auc = roc_auc_score(all_labels, all_probs)
        acc = accuracy_score(all_labels, all_preds)
        f1  = f1_score(all_labels, all_preds, zero_division=0)
        print(f"Target: {target:10s} | AUC: {auc:.4f} | Acc: {acc*100:.2f}% | F1: {f1:.4f}")
        records.append({"Trained on": args.source, "Tested on": target, "AUC": auc, "Accuracy": acc, "F1": f1})

    out_csv = f"zero_shot_results_{args.source}.csv"
    pd.DataFrame(records).to_csv(out_csv, index=False)
    print(f"Saved evaluation results to {out_csv}")

if __name__ == "__main__":
    main()
