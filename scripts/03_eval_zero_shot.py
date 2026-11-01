import os
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from configs.dataset_config import DATASETS, DATASET_NAMES
from src.models_zero_shot import ConvNeXtClassifier

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="Kermany", help="Source dataset name")
    args = parser.parse_args()
    print(f"Evaluating zero-shot transfer for model trained on {args.source} across all target domains...")

if __name__ == "__main__":
    main()
