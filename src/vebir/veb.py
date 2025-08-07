import numpy as np
import cvxpy as cp
from scipy.special import erf
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from tqdm import tqdm


def norm_cdf(z):
    """
    Calculates the cumulative distribution function (CDF) of the standard normal distribution for a given value.

    Parameters
    ----------
    z : float or array-like
        The value(s) at which to evaluate the standard normal CDF.

    Returns
    -------
    float or ndarray
        The CDF value(s) corresponding to the input z.

    Notes
    -----
    This function uses the error function `erf` for computation.
    """
    return 0.5 * (1 + erf(z / np.sqrt(2)))


def norm_pdf(z):
    """
    Compute the value of the standard normal probability density function (PDF) at a given point.

    Parameters
    ----------
    z : float or array-like
        The point(s) at which to evaluate the standard normal PDF.

    Returns
    -------
    float or ndarray
        The value(s) of the standard normal PDF at the specified point(s).
    """
    return np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)


def veb_als_elbo(y, tau, nu, d, mu, W):
    """
    Computes the Evidence Lower Bound (ELBO) to the model evidence for the model y = mu + Wx + r, where x consists of
    iid standard normal random variables and r of iid random variables distributed according the the Gibbs distribution
    associated with the asymmetrically weighted least squares loss. The variational approximation for x|y consists of
    independent, but not identical, normal random variables.
    Parameters
    ----------
    y : np.ndarray
        Observed spectrum of shape (p,).
    tau : float
        Asymmetric loss parameter in (0, 1).
    nu : np.ndarray
        Means of the components of the variational approximation of shape (c,).
    d : np.ndarray
        Standard deviation of the components of the variational approximation of shape (c,).
    mu : np.ndarray
        Mean interference spectrum of shape (p,).
    W : np.ndarray
       First c scaled right singular vectors obtained from a SVD of the centered intereference examples
       of shape (p, c).
    Returns
    -------
    float
        The computed ELBO value.
    """
    p, c = y.shape[0], W.shape[1]

    theta = y - mu - W @ nu
    delta = np.sqrt(np.sum((W * d) ** 2, axis=1))
    z = theta / delta

    g = (theta**2 + delta**2) * (1 - tau) + (2 * tau - 1) * (
        delta * theta * norm_pdf(z) + (theta**2 + delta**2) * norm_cdf(z)
    )
    sigma_hat = 2 * np.mean(g)

    return (
        (c - p) / 2
        + p * np.log(4 / np.pi) / 2
        - p * np.log(sigma_hat) / 2
        + p * np.log(tau) / 2
        - p * np.log(1 + np.sqrt(tau / (1 - tau)))
        - 0.5 * np.sum(nu**2)
        - 0.5 * np.sum(d**2)
        + np.sum(np.log(d))
    )


def veb_als_jac_elbo(y, tau, nu, d, mu, W):
    """
    Computes the gradients (Jacobian) of the ELBO (Evidence Lower Bound) for the VEB-ALS model.
    Parameters
    ----------
    y : np.ndarray
        Observed spectrum of shape (p,).
    tau : float
        Asymmetric loss parameter in (0, 1).
    nu : np.ndarray
        Means of the components of the variational approximation of shape (c,).
    d : np.ndarray
        Standard deviation of the components of the variational approximation of shape (c,).
    mu : np.ndarray
        Mean interference spectrum of shape (p,).
    W : np.ndarray
       First c scaled right singular vectors from the centered intereference examples
       of shape (p, c).
    Returns
    -------
    dtau : float
        Gradient of the ELBO with respect to tau.
    dnu : np.ndarray
        Gradient of the ELBO with respect to nu, shape (c,).
    dd : np.ndarray
        Gradient of the ELBO with respect to d, shape (c,).
    """
    p = y.shape[0]

    theta = y - mu - W @ nu
    delta = np.sqrt(np.sum((W * d) ** 2, axis=1))
    z = theta / delta

    Phi = norm_cdf(z)
    phi = norm_pdf(z)

    g = (theta**2 + delta**2) * (1 - tau) + (2 * tau - 1) * (
        delta * theta * phi + (theta**2 + delta**2) * Phi
    )
    sigma_hat = 2 * np.mean(g)

    a = np.sqrt(tau / (1 - tau))

    dtau = (
        -np.sum(2 * delta * theta * phi + (theta**2 + delta**2) * (2 * Phi - 1))
        / sigma_hat
        + p / tau / 2
        - p / (a * (1 + a) * (1 - tau) ** 2) / 2
    )
    dnu = (
        W.T @ (2 * theta * (1 - tau) + 2 * (2 * tau - 1) * (delta * phi + theta * Phi))
    ) / sigma_hat - nu
    dd = 1 / d - d - 2 * ((W**2).T @ (1 - tau + (2 * tau - 1) * Phi)) * (d / sigma_hat)

    return dtau, dnu, dd


def veb_als_optim_prep(
    y, tau_init, nu_init, d_init, mu, W, tau_min=1e-5, tau_max=0.25, sd_min=1e-10
):
    """
    Prepares the objective function, its Jacobian, initial parameters, and bounds for optimization
    of the VEB-ALS model.
    Args:
        y (np.ndarray): Observed spectrum of shape (p,).
        tau_init (float): Initial value for the tau parameter.
        nu_init (np.ndarray): Initial values for the means of the variational approximation of shape (c,).
        d_init (np.ndarray): Initial values for the standard deviations of the variational approximation of shape (c,).
        mu (np.ndarray): Mean interference spectrum of shape (p,).
        W (np.ndarray): First c scaled right singular vectors from the centered intereference examples of shape (p, c).
        tau_min (float, optional): Minimum bound for tau. Default is 1e-5.
        tau_max (float, optional): Maximum bound for tau. Default is 0.25.
        sd_min (float, optional): Minimum bound for d parameters. Default is 1e-10.
    Returns:
        fun (callable): Objective function to be minimized (negative ELBO).
        jac_fun (callable): Jacobian of the objective function.
        init (np.ndarray): Initial parameter vector for optimization.
        bnds (tuple): Bounds for each parameter in the optimization.
    """
    c = nu_init.shape[0]
    nu_slice = slice(1, 1 + c)
    d_slice = slice(1 + c, None)

    def fun(pars):
        return -veb_als_elbo(y, pars[0], pars[nu_slice], pars[d_slice], mu, W)

    def jac_fun(pars):
        dtau, dnu, dd = veb_als_jac_elbo(
            y, pars[0], pars[nu_slice], pars[d_slice], mu, W
        )
        return -np.concatenate(([dtau], dnu, dd))

    init = np.concatenate(([tau_init], nu_init, d_init))
    bnds = ((tau_min, tau_max),) + ((-np.inf, np.inf),) * c + ((sd_min, np.inf),) * c

    return fun, jac_fun, init, bnds


def als_sigma_hat(y, tau, nu, d, mu, W):
    """
    Computes the estinmated scaling parameter (sigma_hat) of the Gibbs distribution
    associated with asymmetrically weighted least squares loss.
    ----------
    y : np.ndarray
        Observed spectrum of shape (p,).
    tau : float
        Asymmetric loss parameter in (0, 1).
    nu : np.ndarray
        Means of the components of the variational approximation of shape (c,).
    d : np.ndarray
        Standard deviation of the components of the variational approximation of shape (c,).
    mu : np.ndarray
        Mean interference spectrum of shape (p,).
    W : np.ndarray
       First c scaled right singular vectors from the centered intereference examples
       of shape (p, c).
    Returns
    -------
    sigma_hat : float
        Computed sigma_hat value.
    """
    theta = y - mu - W @ nu
    delta = np.sqrt(np.sum((W * d) ** 2, axis=1))
    z = theta / delta

    g = (theta**2 + delta**2) * (1 - tau) + (2 * tau - 1) * (
        delta * theta * norm_pdf(z) + (theta**2 + delta**2) * norm_cdf(z)
    )
    sigma_hat = 2 * np.mean(g)

    return sigma_hat


def als_map(y, mu, W, tau, sigma, mit=100, verbose=False):
    """
    Computes the MAP assingment under the VEB-ALS model.
    Args:
        y (np.ndarray): Observed spectrum of shape (p,).
        mu (np.ndarray): Mean interference spectrum of shape (p,).
        W (np.ndarray): First c scaled right singular vectors from the centered intereference examples of shape (p, c).
        tau (float): Asymmetric loss parameter in (0, 1).
        sigma (float): Scaling parameter of the Gibbs distribution
        mit (int, optional): Maximum number of iterations. Default is 100.
        verbose (bool, optional): If True, displays progress bar and iteration info. Default is False.
    Returns:
        tuple:
            - z (np.ndarray): MAP interference.
            - coef_ (np.ndarray): MAP latent coefficients.
    """
    reg_mod = Ridge(fit_intercept=False, alpha=sigma / 2)

    w = np.repeat(tau, y.shape[0])

    iterator = range(mit)
    iterator = tqdm(iterator) if verbose else iterator

    for i in iterator:
        reg_mod.fit(W, y - mu, sample_weight=w)
        z = mu + reg_mod.predict(W)
        wold = w
        w = np.repeat(tau, y.shape[0])
        w[y - z < 0] = 1 - tau
        if np.all(wold == w):
            break
        if verbose:
            print(i, end=" \r")

    return z, reg_mod.coef_


def veb_pb_elbo(y, tau, nu, d, mu, W):
    """
    Computes the Evidence Lower Bound (ELBO) to the model evidence for the model y = mu + Wx + r, where x consists of
    iid standard normal random variables and r of iid random variables distributed according the the Gibbs distribution
    associated with the pinball loss. The variational approximation for x|y consists of independent, but not identical,
    normal random variables.
    Parameters
    ----------
    y : np.ndarray
        Observed spectrum of shape (p,).
    tau : float
        Asymmetric loss parameter in (0, 1).
    nu : np.ndarray
        Means of the components of the variational approximation of shape (c,).
    d : np.ndarray
        Standard deviation of the components of the variational approximation of shape (c,).
    mu : np.ndarray
        Mean interference spectrum of shape (p,).
    W : np.ndarray
       First c scaled right singular vectors obtained from a SVD of the centered intereference examples
       of shape (p, c).
    Returns
    -------
    float
        The computed ELBO value.
    """
    p, c = y.shape[0], W.shape[1]

    theta = y - mu - W @ nu
    delta = np.sqrt(np.sum((W * d) ** 2, axis=1))
    z = theta / delta

    g = theta * norm_cdf(z) + delta * norm_pdf(z) + (tau - 1) * theta
    sigma_hat = np.mean(g)

    return (
        p * np.log(tau)
        + p * np.log(1 - tau)
        - p * np.log(sigma_hat)
        - 0.5 * np.sum(nu**2)
        - 0.5 * np.sum(d**2)
        + np.sum(np.log(d))
        + (c / 2 - p)
    )


def veb_pb_jac_elbo(y, tau, nu, d, mu, W):
    """
    Computes the gradients (Jacobian) of the ELBO (Evidence Lower Bound) for the VEB-PB model.
    Parameters
    ----------
    y : np.ndarray
        Observed spectrum of shape (p,).
    tau : float
        Asymmetric loss parameter in (0, 1).
    nu : np.ndarray
        Means of the components of the variational approximation of shape (c,).
    d : np.ndarray
        Standard deviation of the components of the variational approximation of shape (c,).
    mu : np.ndarray
        Mean interference spectrum of shape (p,).
    W : np.ndarray
       First c scaled right singular vectors from the centered intereference examples
       of shape (p, c).
    Returns
    -------
    dtau : float
        Gradient of the ELBO with respect to tau.
    dnu : np.ndarray
        Gradient of the ELBO with respect to nu, shape (c,).
    dd : np.ndarray
        Gradient of the ELBO with respect to d, shape (c,).
    """
    p = y.shape[0]

    theta = y - mu - W @ nu
    delta = np.sqrt(np.sum((W * d) ** 2, axis=1))
    z = theta / delta

    Phi = norm_cdf(z)
    phi = norm_pdf(z)
    g = theta * Phi + delta * phi + (tau - 1) * theta
    sigma_hat = np.mean(g)

    dtau = p / tau - p / (1 - tau) - np.sum(theta) / sigma_hat
    dnu = -(W.T @ ((1 - tau) - Phi)) / sigma_hat - nu
    dd = 1 / d - d - ((W**2).T @ (phi / delta)) * (d / sigma_hat)

    return dtau, dnu, dd


def veb_pb_optim_prep(
    y, tau_init, nu_init, d_init, mu, W, tau_min=1e-5, tau_max=0.25, sd_min=1e-10
):
    """
    Prepares the objective function, its Jacobian, initial parameters, and bounds for optimization
    of the VEB-PB model.
    Args:
        y (np.ndarray): Observed spectrum of shape (p,).
        tau_init (float): Initial value for the tau parameter.
        nu_init (np.ndarray): Initial values for the means of the variational approximation of shape (c,).
        d_init (np.ndarray): Initial values for the standard deviations of the variational approximation of shape (c,).
        mu (np.ndarray): Mean interference spectrum of shape (p,).
        W (np.ndarray): First c scaled right singular vectors from the centered intereference examples of shape (p, c).
        tau_min (float, optional): Minimum bound for tau. Default is 1e-5.
        tau_max (float, optional): Maximum bound for tau. Default is 0.25.
        sd_min (float, optional): Minimum bound for d parameters. Default is 1e-10.
    Returns:
        fun (callable): Objective function to be minimized (negative ELBO).
        jac_fun (callable): Jacobian of the objective function.
        init (np.ndarray): Initial parameter vector for optimization.
        bnds (tuple): Bounds for each parameter in the optimization.
    """
    c = nu_init.shape[0]
    nu_slice = slice(1, 1 + c)
    d_slice = slice(1 + c, None)

    def fun(pars):
        return -veb_pb_elbo(y, pars[0], pars[nu_slice], pars[d_slice], mu, W)

    def jac_fun(pars):
        dtau, dnu, dd = veb_pb_jac_elbo(
            y, pars[0], pars[nu_slice], pars[d_slice], mu, W
        )
        return -np.concatenate(([dtau], dnu, dd))

    init = np.concatenate(([tau_init], nu_init, d_init))
    bnds = ((tau_min, tau_max),) + ((-np.inf, np.inf),) * c + ((sd_min, np.inf),) * c

    return fun, jac_fun, init, bnds


def pb_sigma_hat(y, tau, nu, d, mu, W):
    """
    Computes the estinmated scaling parameter (sigma_hat) of the Gibbs distribution
    associated with pinball loss.
    ----------
    y : np.ndarray
        Observed spectrum of shape (p,).
    tau : float
        Asymmetric loss parameter in (0, 1).
    nu : np.ndarray
        Means of the components of the variational approximation of shape (c,).
    d : np.ndarray
        Standard deviation of the components of the variational approximation of shape (c,).
    mu : np.ndarray
        Mean interference spectrum of shape (p,).
    W : np.ndarray
       First c scaled right singular vectors from the centered intereference examples
       of shape (p, c).
    Returns
    -------
    sigma_hat : float
        Computed sigma_hat value.
    """
    theta = y - mu - W @ nu
    delta = np.sqrt(np.sum((W * d) ** 2, axis=1))
    z = theta / delta

    Phi = norm_cdf(z)
    phi = norm_pdf(z)
    g = theta * Phi + delta * phi + (tau - 1) * theta
    sigma_hat = np.mean(g)

    return sigma_hat


def pb_map(y, mu, W, tau, sigma, mit=None, verbose=False):
    """
    Computes the MAP assingment under the VEB-PB model.
    Args:
        y (np.ndarray): Observed spectrum of shape (p,).
        mu (np.ndarray):  Mean interference spectrum of shape (p,).
        W (np.ndarray): First c scaled right singular vectors from the centered intereference examples of shape (p, c).
        tau (float): Asymmetric loss parameter in (0, 1).
        sigma (float): Scaling parameter of the Gibbs distribution
        mit (int, optional): Not used.
        verbose (bool, optional): If True, displays progress bar and iteration info. Default is False.
    Returns:
        tuple:
            - z (np.ndarray): MAP interference.
            - coef_ (np.ndarray): MAP latent coefficients.
    """
    x = cp.Variable(W.shape[1])
    prob = cp.Problem(
        cp.Minimize(
            cp.sum(0.5 * cp.abs(y - mu - W @ x) + (tau - 0.5) * (y - mu - W @ x))
            + 0.5 * sigma * cp.sum_squares(x)
        )
    )

    prob.solve(solver=cp.CLARABEL, verbose=verbose)

    return mu + W @ x.value, x.value


class VEB:
    """
    Variational Empirical Bayes (VEB) model for interference removal.
    Parameters
    ----------
    c : int
        Number of components used to model the intereference/
    tau_init : float, optional
        Initial value for tau parameter. Defaults to 0.1.
    nu_init : array-like, optional
        Initial values for nu parameter. Defaults to zeros of length `c`.
    d_init : array-like, optional
        Initial values for d parameter. Defaults to ones of length `c`.
    loss : str, optional
        Loss function to use. Options are "PB" (default) or "ALS".
    Attributes
    ----------
    tau : float
        Estimated asymmetric loss parameter.
    nu : array-like
        Estimated means of the variational approximation.
    d : array-like
        Estimated standard deviations of the variational approximation.
    sigma_hat : array-like
        Estimated scaling value of the Gibbs distribution.
    interference : array-like
        MAP interference.
    x : array-like
        MAP latent coefficients.
    absorbance : array-like
        MAP absorbance after interference removal.
    Methods
    -------
    fit(y, mu, W, mit=10000, tau_min=1e-5, tau_max=0.25, sd_min=1e-10)
        Fit the VEB model to the data.
    map(y, mu, W, mit)
        Compute the interference and absorbance using the fitted model.
    """

    def __init__(
        self,
        c,
        tau_init=None,
        nu_init=None,
        d_init=None,
        loss="PB",
    ):
        self.loss = loss
        self.c = c

        if tau_init is None:
            self.tau_init = 0.1
        else:
            self.tau_init = tau_init

        if nu_init is None:
            self.nu_init = np.zeros(self.c)
        else:
            self.nu_init = nu_init

        if d_init is None:
            self.d_init = np.ones(self.c)
        else:
            self.d_init = d_init

        if loss == "ALS":
            self.optim_prep = veb_als_optim_prep
            self.comp_sigma_hat = als_sigma_hat
            self.comp_map = als_map

        if loss == "PB":
            self.optim_prep = veb_pb_optim_prep
            self.comp_sigma_hat = pb_sigma_hat
            self.comp_map = pb_map

    def fit(self, y, mu, W, mit=10000, tau_min=1e-5, tau_max=0.25, sd_min=1e-10):
        fun, jac_fun, init, bnds = self.optim_prep(
            y, self.tau_init, self.nu_init, self.d_init, mu, W, tau_min, tau_max, sd_min
        )
        opt = minimize(
            fun,
            init,
            method="L-BFGS-B",
            jac=jac_fun,
            bounds=bnds,
            options={"maxiter": mit},
        )

        self.tau, self.nu, t0 = opt.x[0], opt.x[1 : (1 + self.c)], self.c + 1
        self.d = opt.x[t0:]

        self.sigma_hat = self.comp_sigma_hat(y, self.tau, self.nu, self.d, mu, W)

    def map(self, y, mu, W, mit=100):
        self.interference, self.x = self.comp_map(
            y,
            mu,
            W,
            self.tau,
            self.sigma_hat,
            mit,
            verbose=False,
        )
        self.absorbance = y - self.interference
