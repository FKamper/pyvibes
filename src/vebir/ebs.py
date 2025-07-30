import numpy as np
import cvxpy as cp
from sklearn.linear_model import LinearRegression


def ebs_als(y, V, tau=0.1, mit=100, verbose=False):
    """
    Reconstructs the interference present in a spectra.
    Given a spectrum `y` and a (scaled) right singular vector matrix `V`, this function finds the coefficient vector `x`
    that minimizes the asymmetrically weighted least squares (ALS) loss function charaterized by `tau`. The optimization
    is performed iteratively until convergence or until the maximum number of iterations is reached.
    Args:
        y (np.ndarray): Observation vector of shape (n_wavenumbers,).
        V (np.ndarray): Right singular vector matrix (possibly scaled) of shape (n_wavenumbers, n_components).
        tau (float, optional): Parameter controlling the linear penalty term. Default is 0.1.
        verbose (bool, optional): If True, prints solver output. Default is False.
    Returns:
        tuple:
            - np.ndarray: The reconstructed interference.
            - np.ndarray: The latent loadings `x`, shape (n_wavenumbers,).
    """

    reg_mod = LinearRegression(fit_intercept=False)
    w = np.repeat(tau, y.shape[0])

    for i in range(mit):
        reg_mod.fit(V, y, sample_weight=w)
        z = reg_mod.predict(V)
        wold = w
        w = np.repeat(tau, y.shape[0])
        w[y - z < 0] = 1 - tau
        if np.all(wold == w):
            break
        if verbose:
            print(i, end=" \r")

    return z, reg_mod.coef_


def ebs_pb(y, V, tau=0.1, verbose=False):
    """
    Reconstructs the interference present in a spectrum.
    Given a spectrum `y` and a (scaled) right singular vector matrix `V`, this function finds the coefficient vector `x`
    that minimizes the pinball loss function charaterized by `tau`. The optimization
    is performed using convex optimization with the CLARABEL solver.
    Args:
        y (np.ndarray): Observation vector of shape (n_wavenumbers,).
        V (np.ndarray): Right singular vector matrix (possibly scaled) of shape (n_wavenumbers, n_components).
        tau (float, optional): Parameter controlling the linear penalty term. Default is 0.1.
        verbose (bool, optional): If True, prints solver output. Default is False.
    Returns:
        tuple:
            - np.ndarray: The reconstructed interference.
            - np.ndarray: The latent loadings `x`, shape (n_wavenumbers,).
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
    Parameters
    ----------
    tau : float, optional
        Parameter controlling the asymmetry of the loss function (default is 0.1).
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
        Estimated weights or coefficients from the background estimation.
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
            Mean interference spectrum. Set to zero if centering is not applied.
        W : ndarray
            Scaled right singular vector matrix W = VD obtained from a SVD X - mu = UDV' of the interference examples. Take W = V if no scaling is applied.
        Updates
        -------
        self.x : ndarray
            Estimated latent loading coefficients.
        self.interference : ndarray
            Estimated interference spectrum.
        self.absorbance : ndarray
            Estimated absorbance spectrum.
    """

    def __init__(self, tau=0.1, loss="PB", mit=100):
        self.tau = tau
        self.mit = mit
        self.loss = loss

    def fit(self, y, mu, W):
        if self.loss == "ALS":
            z, x = ebs_als(y - mu, W, tau=self.tau, mit=self.mit, verbose=False)
            self.x = x
            self.interference = mu + z
            self.absorbance = y - self.interference

        if self.loss == "PB":
            z, x = ebs_pb(y - mu, W, tau=self.tau, verbose=False)
            self.x = x
            self.interference = mu + z
            self.absorbance = y - self.interference
