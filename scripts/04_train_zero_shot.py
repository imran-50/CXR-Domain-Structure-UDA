import os
import argparse
import shutil
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score

from configs.dataset_config import DATASETS, CLASS_WEIGHT_RULES

SEED = 42
def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transforms = T.Compose([
    T.Resize((224, 224)),
    T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_transforms = T.Compose([
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

def build_convnext_model():
    model = timm.create_model('convnextv2_tiny', pretrained=True, num_classes=2, in_chans=3)
    n_features = model.head.fc.in_features
    model.head.fc = nn.Sequential(
        nn.Linear(n_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 2)
    )
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="Kermany", choices=list(DATASETS.keys()))
    parser.add_argument("--epochs_head", type=int, default=5)
    parser.add_argument("--epochs_finetune", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    data_dir = os.path.join(DATASETS[args.source], "train")
    image_paths = glob(os.path.join(data_dir, "*/*.jpeg")) + glob(os.path.join(data_dir, "*/*.jpg")) + glob(os.path.join(data_dir, "*/*.png"))
    df = pd.DataFrame({
        'image_path': image_paths,
        'label': [os.path.basename(os.path.dirname(p)) for p in image_paths]
    })

    # Train / Val split (80/20)
    train_df = df.sample(frac=0.8, random_state=SEED)
    val_df = df.drop(train_df.index)

    # Compute asymmetric class weights
    counts = train_df['label'].value_counts()
    rules = CLASS_WEIGHT_RULES.get(args.source, {"normal_multiplier": 1.0, "pneumonia_multiplier": 1.0})
    w_norm = (len(train_df) / (2.0 * counts['NORMAL'])) * rules["normal_multiplier"]
    w_pneu = (len(train_df) / (2.0 * counts['PNEUMONIA'])) * rules["pneumonia_multiplier"]
    weights_tensor = torch.tensor([w_norm, w_pneu], dtype=torch.float).to(device)

    train_loader = DataLoader(PneumoniaDataset(train_df, train_transforms), batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader   = DataLoader(PneumoniaDataset(val_df, val_transforms), batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_convnext_model().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=0.1)

    # Phase 1: Train Head Only
    for p in model.parameters(): p.requires_grad = False
    for p in model.head.fc.parameters(): p.requires_grad = True

    opt_head = optim.AdamW(model.head.fc.parameters(), lr=1e-3, weight_decay=1e-3)
    best_f1 = 0.0

    print(f"--- Phase 1: Training Classifier Head on {args.source} ---")
    for epoch in range(args.epochs_head):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt_head.zero_grad(); loss = criterion(model(x), y); loss.backward(); opt_head.step()

    # Phase 2: Fine-Tuning Full Model
    print(f"--- Phase 2: Fine-Tuning Full Model on {args.source} ---")
    for p in model.parameters(): p.requires_grad = True
    opt_fine = optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if 'head' not in n], 'lr': 1e-5},
        {'params': model.head.fc.parameters(), 'lr': 1e-4}
    ], weight_decay=1e-2)

    for epoch in range(args.epochs_finetune):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt_fine.zero_grad(); loss = criterion(model(x), y); loss.backward(); opt_fine.step()

    out_ckpt = f"best_model_{args.source}_finetuned.pth"
    torch.save(model.state_dict(), out_ckpt)
    print(f"Training Complete! Checkpoint saved to {out_ckpt}")

if __name__ == "__main__":
    main()
