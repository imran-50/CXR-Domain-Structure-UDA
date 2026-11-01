import os
import sys
import glob
import argparse
import itertools
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata

DATASETS = ["Kermany", "PRF", "NIH-RSNA", "CheXpert", "PadChest"]
NAME_MAP = {"NIH": "NIH-RSNA", "Sylhet": "PRF"}
N = len(DATASETS)

PERMS = np.array(list(itertools.permutations(range(N))))
UNORD = list(itertools.combinations(range(N), 2))
IA = np.array([a for a, b in UNORD]); IB = np.array([b for a, b in UNORD])
ORD = [(s, t) for s in range(N) for t in range(N) if s != t]
IS = np.array([s for s, t in ORD]); IT = np.array([t for s, t in ORD])
EPS = 1e-12

def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    xc, yc = x - x.mean(), y - y.mean()
    den = np.sqrt((xc @ xc) * (yc @ yc))
    return float(xc @ yc / den) if den > 0 else float("nan")

def spearman(x, y):
    return pearson(rankdata(x), rankdata(y))

def perm_pvalues(struct, build_x, y):
    struct = np.asarray(struct, float)
    r_obs = pearson(build_x(struct), y)
    rho_obs = spearman(build_x(struct), y)
    c_r = c_rho = 0
    for p in PERMS:
        xs = build_x(struct[p])
        if abs(pearson(xs, y)) >= abs(r_obs) - EPS:
            c_r += 1
        if abs(spearman(xs, y)) >= abs(rho_obs) - EPS:
            c_rho += 1
    return c_r / len(PERMS), c_rho / len(PERMS)

def full_stats(struct, build_x, y):
    x = build_x(np.asarray(struct, float))
    r = pearson(x, y)
    rho = spearman(x, y)
    p_t = float(stats.pearsonr(x, y)[1])
    p_t_rho = float(stats.spearmanr(x, y)[1])
    p_perm_r, p_perm_rho = perm_pvalues(struct, build_x, y)
    return dict(r=r, p_t=p_t, p_perm=p_perm_r, rho=rho, p_rho_t=p_t_rho, p_rho_perm=p_perm_rho)

IDENT = lambda s: s
def X_DIFF(s): return s[IA] - s[IB]
def X_ABSGAP(s): return np.abs(s[IA] - s[IB])
def X_TP(s): return s[IT] - s[IS]

def norm_name(x):
    x = str(x).replace("(in-domain)", "").strip()
    return NAME_MAP.get(x, x)

def load_auc_matrix(path):
    df = pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path)
    src = next(c for c in ("Trained on", "Source") if c in df.columns)
    tgt = next(c for c in ("Tested on", "Target") if c in df.columns)
    M = np.full((N, N), np.nan)
    for _, row in df.iterrows():
        s, t = norm_name(row[src]), norm_name(row[tgt])
        if s == t: continue
        if s in DATASETS and t in DATASETS:
            M[DATASETS.index(s), DATASETS.index(t)] = float(row["AUC"])
    return M

def load_descriptors(csv_path):
    df = pd.read_csv(csv_path)
    df["Dataset"] = df["Dataset"].map(norm_name)
    df = df.set_index("Dataset").loc[DATASETS]
    desc = {
        "Silhouette": df["Silhouette"].to_numpy(float),
        "Fisher Ratio": df["Fisher_Ratio"].to_numpy(float),
        "Davies-Bouldin": df["Davies_Bouldin"].to_numpy(float),
    }
    z = lambda v: (v - v.mean()) / v.std(ddof=1)
    desc["Cluster Score"] = z(desc["Silhouette"]) + z(desc["Fisher Ratio"]) - z(desc["Davies-Bouldin"])
    desc["GBC"] = df["GBC"].to_numpy(float)
    return desc, df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zero-shot", required=True)
    ap.add_argument("--dann", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--csv-robust", required=True)
    args = ap.parse_args()

    DT = load_auc_matrix(args.zero_shot)
    DANN = load_auc_matrix(args.dann)
    desc, df = load_descriptors(args.csv)

    print("\n" + "=" * 80)
    print("TABLE II -- Structural Descriptors")
    print("=" * 80)
    print(f"{'Dataset':10s}{'Sil':>8s}{'DB':>8s}{'Fisher':>8s}{'Cluster':>9s}{'GBC':>8s}")
    for i, d in enumerate(DATASETS):
        print(f"{d:10s}{desc['Silhouette'][i]:8.4f}{desc['Davies-Bouldin'][i]:8.4f}{desc['Fisher Ratio'][i]:8.4f}{desc['Cluster Score'][i]:+9.3f}{desc['GBC'][i]:8.2f}")

    print("\n" + "=" * 80)
    print("TABLE III -- Target Structure vs Incoming AUC (n=5)")
    print("=" * 80)
    incoming = {"Direct Transfer": np.array([np.nanmean(DT[:, t]) for t in range(N)]),
                "DANN": np.array([np.nanmean(DANN[:, t]) for t in range(N)])}
    for rg, y in incoming.items():
        st = full_stats(desc["Silhouette"], IDENT, y)
        print(f"{rg:16s} Silhouette: r={st['r']:+.3f} (p_perm={st['p_perm']:.4f}), rho={st['rho']:+.3f} (p_exact={st['p_rho_perm']:.4f})")

    print("\n" + "=" * 80)
    print("TABLE V -- Signed Difference vs Signed AUC Difference (n=10 pairs)")
    print("=" * 80)
    for rg, M in [("Direct Transfer", DT), ("DANN", DANN)]:
        dA = M[IA, IB] - M[IB, IA]
        st = full_stats(desc["Silhouette"], X_DIFF, dA)
        print(f"{rg:16s} Delta Silhouette: r={st['r']:+.3f} (p_perm={st['p_perm']:.4f})")

    print("\n" + "=" * 80)
    print("TABLE VII -- Transfer Potential vs Directed AUC (n=20 pairs)")
    print("=" * 80)
    for rg, M in [("Direct Transfer", DT), ("DANN", DANN)]:
        dirA = M[IS, IT]
        st = full_stats(desc["Cluster Score"], X_TP, dirA)
        print(f"{rg:16s} TP: r={st['r']:+.3f} (p_perm={st['p_perm']:.4f}, R2={st['r']**2:.3f})")

if __name__ == "__main__":
    main()
