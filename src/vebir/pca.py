from scipy.linalg import svd
import numpy as np


def pca(Z, detrend=True):
    mu = np.mean(Z, axis=0)

    if not detrend:
        mu = 0 * mu

    _, lam, V = svd(Z - mu, full_matrices=False)
    V = V.T
    W = (V * lam) / np.sqrt(Z.shape[0])

    return mu, lam, V, W


def pca_scan(Z, Zte, max_ncomp=53, detrend=True):
    mu = np.mean(Z, axis=0)

    if not detrend:
        mu = 0 * mu

    _, lam, V = svd(Z - mu, full_matrices=False)
    V = V.T

    Err = []
    for i in range(max_ncomp):
        Zhat = mu + (Zte - mu) @ V[:, : (i + 1)] @ V[:, : (i + 1)].T
        Err.append(np.mean((Zte - Zhat) ** 2))

    return np.array(Err)


def loo_pca(Z, max_ncomp=53, detrend=True):
    Err = []
    for i in range(Z.shape[0]):
        Err.append(
            pca_scan(
                np.delete(Z, i, axis=0), Z[[i], :], max_ncomp=max_ncomp, detrend=detrend
            )
        )

    Err = np.array(Err)

    m = np.mean(Err, axis=0)
    se = np.std(Err, axis=0) / np.sqrt(Err.shape[0])

    return 1 + np.argmin(m), 1 + np.min(np.where(m - se <= np.min(m))[0]), Err


def malinowski_ind(Z):
    """
    Computes the IND function exactly as per Equation (6) in
    Malinowski, Anal. Chem. 49 (1977) 612.
    """
    n, m = Z.shape
    k_max = min(n, m)

    # Eigenvalues from SVD
    eigvals = np.linalg.svd(Z, compute_uv=False) ** 2

    ind_values = []
    for j in range(1, k_max):
        residual_sum = np.sum(eigvals[j:])
        RE_j = np.sqrt(residual_sum / (n * (m - j)))
        IND_j = RE_j / (m - j) ** 2
        ind_values.append(IND_j)

    return 1 + np.argmin(ind_values), np.array(ind_values)
