import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from configs.dataset_config import DATASETS, DATASET_NAMES
from src.models_dann import DANN

def main():
    print(f"Configured for 5 datasets: {DATASET_NAMES}")
    print("Run `python scripts/04_train_eval_dann.py` on GPU to adapt across all 20 directed pairs.")

if __name__ == "__main__":
    main()
