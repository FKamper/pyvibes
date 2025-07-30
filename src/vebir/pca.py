from scipy.linalg import svd
import numpy as np


def pca(Z, detrend=True):
    """
    Performs Principal Component Analysis (PCA) on the input data matrix Z.
    Parameters:
        Z (np.ndarray): Input data matrix of shape (n_samples, n_wavenumbers).
        detrend (bool, optional): If True (default), subtracts the mean from each feature (column) before PCA.
                                  If False, the mean is not subtracted.
    Returns:
        mu (np.ndarray): Mean vector of the input data of shape (n_wavenumbers,).
        lam (np.ndarray): Singular values of the input data matrix of shape (min(n_wavenumbers, n_samples),).
        V (np.ndarray): Right singular vales matrix of shape (n_samples, min(n_wavenumbers,n_samples)).
        W (np.ndarray): Scaled right singular vales matrix of shape  (n_samples, min(n_wavenumbers,n_samples)).
    """
    mu = np.mean(Z, axis=0)

    if not detrend:
        mu = 0 * mu

    _, lam, V = svd(Z - mu, full_matrices=False)
    V = V.T
    W = (V * lam) / np.sqrt(Z.shape[0])

    return mu, lam, V, W


def pca_scan(Z, Zte, max_ncomp=53, detrend=True):
    """
    Estimate PCA reconstruction error for varying numbers of principal components using a test set.
    Parameters
    ----------
    Z : np.ndarray
        Training data matrix of shape (n_samples, n_wavenumbers).
    Zte : np.ndarray
        Test data matrix of shape (m_samples, n_wavenumbers).
    max_ncomp : int, optional
        Maximum number of principal components to consider (default is 53).
    detrend : bool, optional
        If True, subtracts the mean of the training data from both training and test data (default is True).
    Returns
    -------
    np.ndarray
        Array of mean squared reconstruction errors for each number of principal components from 1 to max_ncomp.
    Notes
    -----
    Uses singular value decomposition (SVD) to compute principal components. For each number of components,
    projects the test data onto the principal subspace and computes the mean squared error of reconstruction.
    """
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
    """
    Performs leave-one-out (LOO) cross-validation to select the optimal number of principal components for PCA.
    Parameters:
        Z (np.ndarray): Input data matrix of shape (n_samples, n_wavenumbers).
        max_ncomp (int, optional): Maximum number of principal components to consider. Default is 53.
        detrend (bool, optional): Whether to detrend the data before PCA. Default is True.
    Returns:
        chat_min (int): The number of components that minimizes the mean reconstruction error across LOO folds.
        chat_ose (int): The smallest number of components for which the mean reconstruction error is within one standard error of the minimum.
        Err (np.ndarray): Array of shape (n_samples, max_ncomp) containing the error for each fold and component count.
    Notes:
        This function relies on an external function `pca_scan` to compute the error for a given train/test split and number of components.
    """
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

    chat_min = 1 + np.argmin(m)
    chat_ose = 1 + np.min(np.where(m - se <= np.min(m))[0])

    return chat_min, chat_ose, Err


def malinowski_ind(Z):
    """
    Calculates the Malinowski's Indicator Function (IND) for a given data matrix.
    This function implements Equation (6) from Malinowski, Anal. Chem. 49 (1977) 612,
    to estimate the optimal number of principal components in a dataset using the IND criterion.
    Parameters
    ----------
    Z : np.ndarray
        The input data matrix of shape (n_samples, n_wavenumbers).
    Returns
    -------
    optimal_components : int
        The estimated optimal number of principal components (1-based index).
    ind_values : np.ndarray
        Array of IND values for each possible number of components (length min(n, m) - 1).
    References
    ----------
    Malinowski, E. R. (1977). Determination of the number of factors and the experimental error in a data matrix.
    Analytical Chemistry, 49(4), 612-617.
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
