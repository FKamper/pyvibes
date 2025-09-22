import numpy as np
import cvxpy as cp
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from vebir.veb import pinball_loss, als_loss
from itertools import product


def ebs_als(y, V, tau=0.1, mit=100, verbose=False):
    """
    Estimates the interference present in a spectra using the EBS method under asymmetrically weighted least squares loss.
    The optimization is performed iteratively until convergence or until the maximum number of iterations is reached.

    Args
    ----------
    y : np.ndarray
        Observed spectrum of shape (p,).
    V : np.ndarray
        Leading c right singular vectors obtained from a svd of the intereference examples of shape (p,c).
    tau : float
        Asymmetric loss parameter in (0, 1).
    mit: int
        Maximum allowable number of iterations

    Returns
    ----------
    sigma_hat :
        tuple:
            - np.ndarray: The estimated interference.
            - np.ndarray: The latent loadings.
    Notes
    ----------
    Optionally, one can use the right singular vectors after centering the interferenece examples. In this case
    pass y - mu, mu the mean interference spectrum, instead of y and add mu to the interference afterward. One
    could also scale the right singular vectors by their corresponding singulat values.
    """

    reg_mod = LinearRegression(fit_intercept=False)
    w = np.repeat(tau, y.shape[0])

    iterator = range(mit)
    iterator = tqdm(iterator) if verbose else iterator

    for i in iterator:
        reg_mod.fit(V, y, sample_weight=w)
        z = reg_mod.predict(V)
        wold = w
        w = np.repeat(tau, y.shape[0])
        w[y - z < 0] = 1 - tau
        if np.all(wold == w):
            break

    return z, reg_mod.coef_


def ebs_pb(y, V, tau=0.1, mit=None, verbose=False):
    """
    Estimates the interference present in a spectra using the EBS method under pinball loss.
    The optimization is performed using the CLARABEL solver.

    Args
    ----------
    y : np.ndarray
        Observed spectrum of shape (p,).
    V : np.ndarray
        Leading c right singular vectors obtained from a svd of the intereference examples of shape (p,c).
    tau : float
        Asymmetric loss parameter in (0, 1).
    mit: int
        Maximum allowable number of iterations, not used currently.

    Returns
    ----------
    sigma_hat :
        tuple:
            - np.ndarray: The estimated interference.
            - np.ndarray: The latent loadings.
    Notes
    ----------
    Optionally, one can use the right singular vectors after centering the interferenece examples. In this case
    pass y - mu, mu the mean interference spectrum, instead of y and add mu to the interference afterward. One
    could also scale the right singular vectors by their corresponding singulat values.
    """
    x = cp.Variable(V.shape[1])
    prob = cp.Problem(
        cp.Minimize(cp.sum(0.5 * cp.abs(y - V @ x) + (tau - 0.5) * (y - V @ x)))
    )

    prob.solve(solver=cp.CLARABEL, verbose=verbose)
    return V @ x.value, x.value


class EBS:
    """
    Eliminate Background Spectrum (EBS) algorithm for interference remocal in spectral data.

    Args
    ----------
    tau : float, optional
        Asymmetric loss parameter in (0, 1).
    loss : str, optional
        Loss function to use for background estimation. Options are:
            - "PB": Pinball Loss
            - "ALS": Asymmetric Weighted Least Squares
        Default is "PB".
    mit : int, optional
        Maximum number of iterations for the ALS algorithm (default is 100).

    Attributes
    ----------
    tau : float
        Asymmetric parameter.
    mit : int
        Maximum number of iterations.
    loss : str
        Selected loss function.
    x : ndarray
        Estimated latent interference loadings.
    interference : ndarray
        Estimated interference spectrum.
    absorbance : ndarray
        Estimated absorbance spectrum.

    Methods
    -------
    fit(y, mu, W)
        Estimate and remove intereference from the observed spectrum.
        Parameters
        ----------
        y : ndarray
            Observed spectrum.
        mu : ndarray
            Mean interference spectrum. Set to zero if centering is not applied. Shape (p,c).
        W : ndarray
            Scaled/unscaled right singular vectors obtained from a svd of the intereference examples of shape (p,c).
        Updates
        -------
        self.x : ndarray
            Estimated latent loading coefficients.
        self.interference : ndarray
            Estimated interference spectrum.
        self.absorbance : ndarray
            Estimated absorbance spectrum.
    """

    def __init__(self, tau=0.1, loss="PB"):
        self.tau = tau
        self.loss = loss

        if self.loss == "PB":
            self.comp_map = ebs_pb

        if self.loss == "ALS":
            self.comp_map = ebs_als

    def fit(self, y, mu, W, mit=100):
        self.interference, self.x = self.comp_map(
            y - mu,
            W,
            self.tau,
            mit,
            verbose=False,
        )
        self.interference = mu + self.interference
        self.absorbance = y - self.interference


class EbsCV:
    def __init__(
        self,
        y,
        mu,
        W,
        loss="PB",
        tau_grid=[0.1],
        c_grid=[10],
        num_folds=5,
    ):
        self.loss = loss
        self.num_folds = num_folds

        self.tau_grid = tau_grid
        self.c_grid = c_grid
        self.y = y
        self.mu = mu
        self.W = W
        self.folds = np.array_split(np.arange(y.shape[0]), num_folds)

        if self.loss == "ALS":
            self.loss_fun = als_loss
            self.interference_estimator = ebs_als
        if self.loss == "PB":
            self.loss_fun = pinball_loss
            self.interference_estimator = ebs_pb

    def compute_cv_errors(self, verbose=False):
        self.cv_errs = np.zeros((len(self.tau_grid), len(self.c_grid)))
        idx_pairs = list(
            product(range(self.cv_errs.shape[0]), range(self.cv_errs.shape[1]))
        )

        for i, j in tqdm(idx_pairs, disable=not verbose):
            a = np.zeros(self.y.shape[0])
            tau = self.tau_grid[i]
            c = self.c_grid[j]

            for fold in self.folds:
                keep_idx = np.concatenate([f for f in self.folds if f is not fold])
                _, x = self.interference_estimator(
                    (self.y - self.mu)[keep_idx], self.W[keep_idx, :c], tau=tau
                )
                a[fold] = self.y[fold] - (self.mu[fold] + self.W[fold, :c] @ x)

            self.cv_errs[i, j] = np.sum(self.loss_fun(a, tau=tau))

        row_idx, col_idx = np.unravel_index(np.argmin(self.cv_errs), self.cv_errs.shape)
        self.opt_tau = self.tau_grid[row_idx]
        self.opt_c = self.c_grid[col_idx]

    def estimate_absorbance(self, mit=100):
        self.z, self.x = self.interference_estimator(
            (self.y - self.mu), self.W[:, : self.opt_c], tau=self.opt_tau, mit=mit
        )
        self.interference = self.mu + self.z
        self.absorbance = self.y - self.interference
