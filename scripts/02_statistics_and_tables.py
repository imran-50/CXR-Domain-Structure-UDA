import os
import argparse
import itertools
import numpy as np
import pandas as pd
from scipy.stats import rankdata, binomtest

DATASETS = ['Kermany', 'PRF', 'NIH-RSNA', 'CheXpert', 'PadChest']
NAME_MAP = {'NIH': 'NIH-RSNA', 'Sylhet': 'PRF'}
REGIMES = ['DANN', 'Direct transfer']

DESCRIPTORS = [
    ('Silhouette', 'Silhouette score'),
    ('Fisher_Ratio', 'Fisher Ratio')
]

BENCHMARK = [
    ('Silhouette',     'Silhouette score'),
    ('Fisher_Ratio',   'Fisher Ratio'),
    ('GBC',            'GBC (CVPR 2022)'),
    ('LogME',          'LogME (ICML 2021)'),
    ('H_score',        'H-score (ICIP 2019)'),
    ('H_score_shrunk', 'H-score shrunk (2022)'),
    ('TransRate',      'TransRate (ICML 2022)'),
    ('NLEEP',          'NLEEP (CVPR 2021)'),
    ('PACTran_Gauss',  'PACTran-Gauss (ECCV 2022)'),
    ('SFDA',           'SFDA (ECCV 2022)'),
]

TABLE3 = [('Silhouette', 'Silhouette'), ('Fisher_Ratio', 'Fisher'), ('GBC', 'GBC'),
          ('LogME', 'LogME'), ('H_score_shrunk', 'H-score (s)'),
          ('TransRate', 'TransRate'), ('NLEEP', 'NLEEP'),
          ('PACTran_Gauss', 'PACTran-G'), ('SFDA', 'SFDA')]

PERMS = list(itertools.permutations(range(5)))
UNORDERED = list(itertools.combinations(range(5), 2))
IA = np.array([a for a, _ in UNORDERED]); IB = np.array([b for _, b in UNORDERED])
ORDERED = [(s, t) for s in range(5) for t in range(5) if s != t]
IS = np.array([s for s, _ in ORDERED]); IT = np.array([t for _, t in ORDERED])

IDENT  = lambda v: v
SIGNED = lambda v: v[IA] - v[IB]
ABSGAP = lambda v: np.abs(v[IA] - v[IB])
TPOT   = lambda v: v[IT] - v[IS]

def norm(x):
    x = str(x).replace('(in-domain)', '').strip()
    return NAME_MAP.get(x, x)

def auc_matrix(path, src_col, tgt_col):
    df = pd.read_csv(path) if path.endswith('.csv') else pd.read_excel(path)
    M = np.full((5, 5), np.nan)
    for _, r in df.iterrows():
        s, t = norm(r[src_col]), norm(r[tgt_col])
        if s != t and s in DATASETS and t in DATASETS:
            M[DATASETS.index(s), DATASETS.index(t)] = float(r['AUC'])
    missing = [(DATASETS[i], DATASETS[j]) for i in range(5) for j in range(5)
               if i != j and np.isnan(M[i, j])]
    if missing:
        raise ValueError(f'{path}: missing directed pairs {missing}')
    return M

def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    xc, yc = x - x.mean(), y - y.mean()
    den = np.sqrt((xc @ xc) * (yc @ yc))
    return float(xc @ yc / den) if den > 0 else np.nan

def spearman(x, y):
    return pearson(rankdata(x), rankdata(y))

def perm_stats(values, build, y):
    v = np.asarray(values, float)
    r_obs, rho_obs = pearson(build(v), y), spearman(build(v), y)
    cr = cs = 0
    for q in PERMS:
        xs = build(v[list(q)])
        if abs(pearson(xs, y)) >= abs(r_obs) - 1e-12: cr += 1
        if abs(spearman(xs, y)) >= abs(rho_obs) - 1e-12: cs += 1
    return {'r': r_obs, 'p_r': cr / 120.0, 'rho': rho_obs, 'p_rho': cs / 120.0}

def head(t):
    print('\n' + '=' * 104); print(t); print('=' * 104)

def two_regime_table(title, measures, desc, build, Y, latex_label, latex_caption):
    head(title)
    print(f"{ 'Measure':27s}|{'DANN':^36s}|{'Direct transfer':^36s}")
    print(f"{ '':27s}|{'r':>8s}{'p':>9s}{'rho':>9s}{'p_rho':>9s} |{'r':>8s}{'p':>9s}{'rho':>9s}{'p_rho':>9s}")
    print('-' * 104)
    rows = []
    for col, name in measures:
        v = desc[col].to_numpy(float)
        sd = perm_stats(v, build, Y['DANN'])
        st = perm_stats(v, build, Y['Direct transfer'])
        rows.append((name, sd, st))
        print(f"{name:27s}|{sd['r']:>+8.3f}{sd['p_r']:>9.4f}{sd['rho']:>+9.3f}{sd['p_rho']:>9.4f} |{st['r']:>+8.3f}{st['p_r']:>9.4f}{st['rho']:>+9.3f}{st['p_rho']:>9.4f}")
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--primary', default='outputs/feature_space_analysis_nih.csv')
    ap.add_argument('--robust', default='outputs/feature_space_analysis_chex.csv')
    ap.add_argument('--zero-shot', default='data/transfer_results/zero_shot_report_by_target.csv')
    ap.add_argument('--dann', default='data/transfer_results/uda_dann_results_by_target.csv')
    a = ap.parse_args()

    p_csv = a.primary if os.path.exists(a.primary) else 'feature_space_analysis_nih.csv'
    r_csv = a.robust if os.path.exists(a.robust) else 'feature_space_analysis_chex.csv'
    zs_p  = a.zero_shot if os.path.exists(a.zero_shot) else '/kaggle/input/datasets/imran50/001zero-shot-report-by-target/zero_shot_report_by_target.xlsx'
    da_p  = a.dann if os.path.exists(a.dann) else '/kaggle/input/datasets/imran50/002uda-dann-results-by-target/uda_dann_results_by_target.xlsx'

    P = pd.read_csv(p_csv).set_index('Dataset').loc[DATASETS]
    R = pd.read_csv(r_csv).set_index('Dataset').loc[DATASETS]
    DT = auc_matrix(zs_p, 'Trained on', 'Tested on')
    DA = auc_matrix(da_p, 'Source', 'Target')
    M = {'DANN': DA, 'Direct transfer': DT}

    incoming = {k: np.array([np.nanmean(v[:, t]) for t in range(5)]) for k, v in M.items()}
    outgoing = {k: np.array([np.nanmean(v[s, :]) for s in range(5)]) for k, v in M.items()}
    signed   = {k: v[IA, IB] - v[IB, IA] for k, v in M.items()}
    meanpair = {k: (v[IA, IB] + v[IB, IA]) / 2 for k, v in M.items()}
    directed = {k: v[IS, IT] for k, v in M.items()}

    head('MEAN AUC PER DATASET')
    print(f"{ 'Dataset':10s}{'in DANN':>10s}{'in direct':>11s}{'out DANN':>11s}{'out direct':>12s}")
    for i, d in enumerate(DATASETS):
        print(f"{d:10s}{incoming['DANN'][i]:>10.4f}{incoming['Direct transfer'][i]:>11.4f}{outgoing['DANN'][i]:>11.4f}{outgoing['Direct transfer'][i]:>12.4f}")
    print(f"{ 'overall':10s}{np.nanmean(DA):>10.4f}{np.nanmean(DT):>11.4f}")

    head('TABLE III  Descriptors and Reference Estimators per Dataset (Primary Extractor NIH)')
    print(f"{ 'Dataset':10s}" + ''.join(f"{h:>13s}" for _, h in TABLE3))
    for d in DATASETS:
        print(f"{d:10s}" + ''.join(f"{P.loc[d, c]:>13.4f}" for c, _ in TABLE3))

    two_regime_table(
        'TABLE IV  Target Structure vs Mean Incoming AUC (n=5; Exact Permutation p)',
        BENCHMARK, P, IDENT, incoming, 'tab:target',
        r'Target Structure and Reference Transferability Estimators versus Mean Incoming AUC')

    two_regime_table(
        'TABLE V  Source Structure vs Mean Outgoing AUC (n=5; Exact Permutation p)',
        DESCRIPTORS, P, IDENT, outgoing, 'tab:source',
        r'Source Structure versus Mean Outgoing AUC')

    head('TABLE VI  Pair-Level Analyses (n=10 Unordered Pairs; Dataset-Label Permutation p)')
    print(f"{ 'Descriptor':18s}{'Analysis':38s}|{'DANN (r, p)':^22s}|{'Direct transfer (r, p)':^22s}")
    print('-' * 104)
    for col, dname in DESCRIPTORS:
        v = P[col].to_numpy(float)
        for lbl, build, Y in [('Signed diff vs signed dAUC', SIGNED, signed),
                              ('Absolute gap vs mean pair AUC', ABSGAP, meanpair)]:
            sd = perm_stats(v, build, Y['DANN'])
            st = perm_stats(v, build, Y['Direct transfer'])
            print(f"{dname:18s}{lbl:38s}| {sd['r']:>+7.3f} ({sd['p_r']:.4f})     | {st['r']:>+7.3f} ({st['p_r']:.4f})")

    head('TABLE VII  Transfer Potential vs Directed AUC (n=20 Directed Pairs)')
    print(f"{ 'Descriptor':18s}{'Regime':18s}{'r':>9s}{'p':>10s}{'rho':>10s}{'p_rho':>10s}{'R2':>9s}")
    for col, dname in DESCRIPTORS:
        v = P[col].to_numpy(float)
        for rg in REGIMES:
            s = perm_stats(v, TPOT, directed[rg])
            print(f"{dname:18s}{rg:18s}{s['r']:>+9.3f}{s['p_r']:>10.4f}{s['rho']:>+10.3f}{s['p_rho']:>10.4f}{s['r']**2:>9.3f}")

    head('TABLE VIII  Direction Prediction (10 Unordered Pairs)')
    print(f"{ 'Descriptor':18s}{'Regime':18s}{'correct':>10s}{'binomial p':>13s}   misses")
    for col, dname in DESCRIPTORS:
        v = P[col].to_numpy(float)
        for rg in REGIMES:
            A = M[rg]; hits, miss = 0, []
            for i, j in UNORDERED:
                if np.sign(v[j] - v[i]) == np.sign(A[i, j] - A[j, i]):
                    hits += 1
                else:
                    miss.append(f'{DATASETS[i]}/{DATASETS[j]}')
            p = binomtest(hits, 10, 0.5, alternative='greater').pvalue
            print(f"{dname:18s}{rg:18s}{hits:>7d}/10{p:>13.4f}   {', '.join(miss) if miss else '-'}")

    head('TABLE IX  Robustness to the Feature Extractor (NIH Primary vs CheXpert Robustness)')
    print(f"{ 'Descriptor':16s}{'Relationship':32s}{'Regime':16s}{'primary':>10s}{'robustness':>13s}")
    checks = [('Separability -> incoming AUC', IDENT, incoming),
              ('Separability -> outgoing AUC', IDENT, outgoing),
              ('Signed difference -> dAUC',    SIGNED, signed),
              ('Transfer Potential -> AUC',    TPOT,   directed)]
    for col, dname in DESCRIPTORS:
        for lbl, build, Y in checks:
            for rg in REGIMES:
                r1 = pearson(build(P[col].to_numpy(float)), Y[rg])
                r2 = pearson(build(R[col].to_numpy(float)), Y[rg])
                print(f"{dname:16s}{lbl:32s}{rg:16s}{r1:>+10.3f}{r2:>+13.3f}")

if __name__ == '__main__':
    main()
