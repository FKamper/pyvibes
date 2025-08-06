import numpy as np
from scipy.special import erf
import cvxpy as cp
from scipy.optimize import minimize
from sklearn.linear_model import Ridge


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


def veb_ebs_als_elbo(y, tau, nu, d, mu, W):
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


def veb_ebs_als_jac_elbo(y, tau, nu, d, mu, W):
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


def veb_ebs_als_optim_prep(y, tau_init, nu_init, d_init, mu, W, tau_up=0.25):
    c = nu_init.shape[0]
    nu_slice = slice(1, 1 + c)
    d_slice = slice(1 + c, None)

    def fun(pars):
        return -veb_ebs_als_elbo(y, pars[0], pars[nu_slice], pars[d_slice], mu, W)

    def jac_fun(pars):
        dtau, dnu, dd = veb_ebs_als_jac_elbo(
            y, pars[0], pars[nu_slice], pars[d_slice], mu, W
        )
        return -np.concatenate(([dtau], dnu, dd))

    init = np.concatenate(([tau_init], nu_init, d_init))
    bnds = ((0.00001, tau_up),) + ((-np.inf, np.inf),) * c + ((1e-10, np.inf),) * c

    return fun, jac_fun, init, bnds


def als_sigma_hat(y, tau, nu, d, mu, W):
    theta = y - mu - W @ nu
    delta = np.sqrt(np.sum((W * d) ** 2, axis=1))
    z = theta / delta

    g = (theta**2 + delta**2) * (1 - tau) + (2 * tau - 1) * (
        delta * theta * norm_pdf(z) + (theta**2 + delta**2) * norm_cdf(z)
    )
    sigma_hat = 2 * np.mean(g)

    return sigma_hat


def als_map(y, mu, W, tau, sigma, maxit=100, warm_start=None, verbose=False):
    reg_mod = Ridge(fit_intercept=False, alpha=sigma / 2)
    w = np.repeat(tau, y.shape[0])

    for i in range(maxit):
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


def veb_ebs_pb_elbo(y, tau, nu, d, mu, W):
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


def veb_ebs_pb_jac_elbo(y, tau, nu, d, mu, W):
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


def veb_ebs_pb_optim_prep(y, tau_init, nu_init, d_init, mu, W, tau_up=0.25):
    c = nu_init.shape[0]
    nu_slice = slice(1, 1 + c)
    d_slice = slice(1 + c, None)

    def fun(pars):
        return -veb_ebs_pb_elbo(y, pars[0], pars[nu_slice], pars[d_slice], mu, W)

    def jac_fun(pars):
        dtau, dnu, dd = veb_ebs_pb_jac_elbo(
            y, pars[0], pars[nu_slice], pars[d_slice], mu, W
        )
        return -np.concatenate(([dtau], dnu, dd))

    init = np.concatenate(([tau_init], nu_init, d_init))
    bnds = ((0.00001, tau_up),) + ((-np.inf, np.inf),) * c + ((1e-10, np.inf),) * c

    return fun, jac_fun, init, bnds


def pb_sigma_hat(y, tau, nu, d, mu, W):
    theta = y - mu - W @ nu
    delta = np.sqrt(np.sum((W * d) ** 2, axis=1))
    z = theta / delta

    Phi = norm_cdf(z)
    phi = norm_pdf(z)
    g = theta * Phi + delta * phi + (tau - 1) * theta
    sigma_hat = np.mean(g)

    return sigma_hat


def pb_map(y, mu, W, tau, sigma, warm_start=None, verbose=False):
    beta = cp.Variable(W.shape[1])
    prob = cp.Problem(
        cp.Minimize(
            cp.sum(0.5 * cp.abs(y - mu - W @ beta) + (tau - 0.5) * (y - mu - W @ beta))
            + 0.5 * sigma * cp.sum_squares(beta)
        )
    )

    if warm_start is None:
        prob.solve(solver=cp.CLARABEL, verbose=verbose)
    else:
        prob.solve(solver=cp.CLARABEL, verbose=verbose, warm_start=warm_start)

    return mu + W @ beta.value, beta.value


class VEB:
    def __init__(
        self,
        c,
        tau_init=None,
        nu_init=None,
        d_init=None,
        loss="PB",
        mit=10000,
        tau_up=0.25,
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
            self.optim_prep = veb_ebs_als_optim_prep
            self.comp_sigma_hat = als_sigma_hat
            self.comp_map = als_map

        if loss == "PB":
            self.optim_prep = veb_ebs_pb_optim_prep
            self.comp_sigma_hat = pb_sigma_hat
            self.comp_map = pb_map

    def fit(self, y, mu, W, mit=10000, tau_max=0.25):
        fun, jac_fun, init, bnds = self.optim_prep(
            y, self.tau_init, self.nu_init, self.d_init, mu, W, tau_max
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

    def map(self, y, mu, W):
        self.interference, self.x = self.comp_map(
            y,
            mu,
            W,
            self.tau,
            self.sigma_hat,
            warm_start=self.nu,
            verbose=False,
        )
        self.absorbance = y - self.interference
