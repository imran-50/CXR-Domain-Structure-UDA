import os
import json
import copy
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score
from src.models_dann import DANN

DATASETS = {
    'Kermany':  '/kaggle/input/datasets/imran50/kermany-pcxr/chest_xray',
    'PRF':      '/kaggle/input/datasets/imran50/sylhet-train-test/sylhet/chest_xray',
    'NIH-RSNA': '/kaggle/input/datasets/imran50/nih-rsna-acxr',
    'CheXpert': '/kaggle/input/datasets/imran50/chexpert',
    'PadChest': '/kaggle/input/datasets/imran50/padchest-8bit',
}
STUDY_DATASETS = list(DATASETS.keys())
ALL_PAIRS = [(s, t) for s in STUDY_DATASETS for t in STUDY_DATASETS if s != t]

def main():
    print(f'DANN Adversarial Adaptation Pipeline across {len(ALL_PAIRS)} directed pairs.')
    print('Architecture: ResNet-18 (512-d) + GRL + 256-d MLP label & domain classifiers.')

if __name__ == '__main__':
    main()
