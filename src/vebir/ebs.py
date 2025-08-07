import numpy as np
import cvxpy as cp
from tqdm import tqdm
from sklearn.linear_model import LinearRegression


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
