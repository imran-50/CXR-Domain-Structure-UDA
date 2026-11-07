import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

def silhouette(X, y, sample_size=3000, seed=42):
    Xs = StandardScaler().fit_transform(X.astype(np.float64))
    return float(silhouette_score(Xs, y, sample_size=min(sample_size, len(y)), random_state=seed))

def fisher_ratio(X, y, eps=1e-12):
    Xs = StandardScaler().fit_transform(X.astype(np.float64))
    X0, X1 = Xs[y == 0], Xs[y == 1]
    b = float(np.sum((X0.mean(0) - X1.mean(0)) ** 2))
    w = float(np.sum(X0.var(0, ddof=1)) + np.sum(X1.var(0, ddof=1)))
    return b / (w + eps)

def gbc(X, y, n_components=64, seed=42):
    X = np.asarray(X, dtype=np.float64)
    d = min(n_components, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=d, random_state=seed)
    Z = pca.fit_transform(X)
    Z0, Z1 = Z[y == 0], Z[y == 1]
    m0, m1 = Z0.mean(0), Z1.mean(0)
    s0, s1 = max(float(Z0.var(0, ddof=1).mean()), 1e-6), max(float(Z1.var(0, ddof=1).mean()), 1e-6)
    m = 0.5 * (s0 + s1)
    db = float(np.sum((m0 - m1) ** 2) / (8.0 * m) + 0.5 * d * np.log(m / np.sqrt(s0 * s1)))
    ret_var = float(np.sum(pca.explained_variance_ratio_))
    return {'GBC': 2.0 * db, 'D_B': db, 'pca_dim': d, 'retained_var': ret_var}

def logme(X, y):
    X = StandardScaler().fit_transform(X.astype(np.float64))
    y_bin = np.where(y == 0, -1.0, 1.0)
    N, D = X.shape
    u, s, _ = np.linalg.svd(X, full_matrices=False)
    s2 = s ** 2
    Uy = u.T @ y_bin
    Uy2 = Uy ** 2
    alpha, beta = 1.0, 1.0
    for _ in range(50):
        gamma = np.sum(beta * s2 / (alpha + beta * s2))
        m2 = beta * np.sum(s2 * Uy2 / ((alpha + beta * s2) ** 2))
        alpha = max(gamma / max(m2, 1e-12), 1e-6)
        res = np.sum(y_bin ** 2) - np.sum(beta * s2 * Uy2 / (alpha + beta * s2))
        beta = max((N - gamma) / max(res, 1e-12), 1e-6)
    ev = 0.5 * (D * np.log(alpha) + N * np.log(beta) - alpha * m2 - beta * res - np.sum(np.log(alpha + beta * s2)) - N * np.log(2 * np.pi))
    return float(ev / N)

def h_score(X, y, shrunk=False):
    X = StandardScaler().fit_transform(X.astype(np.float64))
    N, D = X.shape
    cov_f = np.cov(X, rowvar=False)
    if shrunk:
        trace = np.trace(cov_f)
        cov_f = 0.8 * cov_f + 0.2 * (trace / D) * np.eye(D)
    mu0, mu1 = X[y == 0].mean(0), X[y == 1].mean(0)
    p0, p1 = np.mean(y == 0), np.mean(y == 1)
    mu_total = p0 * mu0 + p1 * mu1
    cov_cond = p0 * np.outer(mu0 - mu_total, mu0 - mu_total) + p1 * np.outer(mu1 - mu_total, mu1 - mu_total)
    inv_cov = np.linalg.pinv(cov_f)
    return float(np.trace(inv_cov @ cov_cond))

def transrate(X, y, eps=0.5):
    X = StandardScaler().fit_transform(X.astype(np.float64))
    N, D = X.shape
    def rate(Z, n_samples):
        p = D / (n_samples * (eps ** 2))
        return 0.5 * float(np.linalg.slogdet(np.eye(D) + p * (Z.T @ Z))[1] / np.log(2.0))
    rz = rate(X, N)
    rzy = 0.0
    for c in np.unique(y):
        Xc = X[y == c]
        rzy += (len(Xc) / N) * rate(Xc, len(Xc))
    return float(rz - rzy)

def nleep(X, y, seed=42):
    X = StandardScaler().fit_transform(X.astype(np.float64))
    d = min(32, X.shape[1], X.shape[0] - 1)
    Z = PCA(n_components=d, random_state=seed).fit_transform(X)
    Z0, Z1 = Z[y == 0], Z[y == 1]
    m0, m1 = Z0.mean(0), Z1.mean(0)
    s0 = max(float(Z0.var(0).mean()), 1e-4)
    s1 = max(float(Z1.var(0).mean()), 1e-4)
    p0 = np.exp(-0.5 * np.sum((Z - m0) ** 2, axis=1) / s0) / (np.sqrt(s0 ** d))
    p1 = np.exp(-0.5 * np.sum((Z - m1) ** 2, axis=1) / s1) / (np.sqrt(s1 ** d))
    prob1 = np.clip(p1 / (p0 + p1 + 1e-12), 1e-6, 1.0 - 1e-6)
    log_lik = np.where(y == 1, np.log(prob1), np.log(1.0 - prob1))
    return float(np.mean(log_lik))

def pactran(X, y):
    X = StandardScaler().fit_transform(X.astype(np.float64))
    N, D = X.shape
    X0, X1 = X[y == 0], X[y == 1]
    mu_diff = np.linalg.norm(X0.mean(0) - X1.mean(0))
    trace_cov = float(np.trace(np.cov(X, rowvar=False)))
    return float(- (trace_cov / D) - (1.0 / max(mu_diff, 1e-4)))

def sfda(X, y):
    X = StandardScaler().fit_transform(X.astype(np.float64))
    fr = fisher_ratio(X, y)
    return float(np.log(fr + 1e-6) - 1.0)

def compute_all_estimators(X, y, seed=42):
    g = gbc(X, y, n_components=64, seed=seed)
    return {
        'Silhouette': silhouette(X, y, seed=seed),
        'Fisher_Ratio': fisher_ratio(X, y),
        'GBC': g['GBC'],
        'GBC_D_B': g['GBC_D_B'],
        'GBC_pca_dim': g['pca_dim'],
        'GBC_retained_variance': g['retained_var'],
        'LogME': logme(X, y),
        'H_score': h_score(X, y, shrunk=False),
        'H_score_shrunk': h_score(X, y, shrunk=True),
        'TransRate': transrate(X, y),
        'NLEEP': nleep(X, y, seed=seed),
        'PACTran_Gauss': pactran(X, y),
        'SFDA': sfda(X, y),
    }
