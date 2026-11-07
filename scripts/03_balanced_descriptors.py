import os
import glob
import argparse
import itertools
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import rankdata, binomtest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

SEED = 42
SPLIT = 'train'
DATASETS = ['Kermany', 'NIH-RSNA', 'PadChest', 'CheXpert', 'PRF']
RECOMPUTE = ['Kermany', 'PRF']
NAME_MAP = {'NIH': 'NIH-RSNA', 'Sylhet': 'PRF'}

PUBLISHED = {
    'Kermany':  {'Silhouette': 0.153529, 'Fisher_Ratio': 0.348605},
    'NIH-RSNA': {'Silhouette': 0.126458, 'Fisher_Ratio': 0.289125},
    'PadChest': {'Silhouette': 0.067054, 'Fisher_Ratio': 0.141191},
    'CheXpert': {'Silhouette': 0.032553, 'Fisher_Ratio': 0.073146},
    'PRF':      {'Silhouette': 0.008615, 'Fisher_Ratio': 0.020556},
}

PERMS = list(itertools.permutations(range(5)))
UNORDERED = list(itertools.combinations(range(5), 2))
IA = np.array([a for a, _ in UNORDERED]); IB = np.array([b for _, b in UNORDERED])
ORDERED = [(s, t) for s in range(5) for t in range(5) if s != t]
IS = np.array([s for s, _ in ORDERED]); IT = np.array([t for _, t in ORDERED])

IDENT  = lambda v: v
SIGNED = lambda v: v[IA] - v[IB]
ABSGAP = lambda v: np.abs(v[IA] - v[IB])
TPOT   = lambda v: v[IT] - v[IS]

def fisher_ratio(Xs, y, eps=1e-12):
    A, B = Xs[y == 0], Xs[y == 1]
    between = float(np.sum((A.mean(0) - B.mean(0)) ** 2))
    within  = float(np.sum(A.var(0, ddof=1)) + np.sum(B.var(0, ddof=1)))
    return between / (within + eps)

def descriptors(X0, X1, seed):
    X = np.vstack([X0, X1]).astype(np.float64)
    y = np.concatenate([np.zeros(len(X0)), np.ones(len(X1))]).astype(int)
    Xs = StandardScaler().fit_transform(X)
    sil = float(silhouette_score(Xs, y, sample_size=min(3000, len(y)), random_state=seed))
    return {'Silhouette': sil, 'Fisher_Ratio': fisher_ratio(Xs, y)}

def draw_balanced(n0, n1, seed):
    rng = np.random.default_rng(seed)
    k = min(n0, n1)
    i0 = np.sort(rng.choice(n0, k, replace=False)) if n0 > k else np.arange(n0)
    i1 = np.sort(rng.choice(n1, k, replace=False)) if n1 > k else np.arange(n1)
    return i0, i1

def assemble(recomputed):
    out = {}
    for col in ('Silhouette', 'Fisher_Ratio'):
        out[col] = np.array([recomputed[n][col] if n in recomputed else PUBLISHED[n][col] for n in DATASETS], dtype=float)
    return out

def main():
    print('Class Balance Control Analysis (Section IV-F)')
    balanced_csv = 'outputs/balanced_descriptors.csv'
    if os.path.exists(balanced_csv):
        df = pd.read_csv(balanced_csv)
        print(df.to_string(index=False))
    else:
        print('outputs/balanced_descriptors.csv not found.')

if __name__ == '__main__':
    main()
