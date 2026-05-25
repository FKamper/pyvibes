import numpy as np
import cvxpy as cp

from vibes.absorbance_estimators.vibes import norm_pdf, norm_cdf


def veb_pb_elbo(y, tau, nu, d, mu, W):
    p = y.shape[0]

    theta = y - mu - W @ nu
    delta = np.sqrt(np.sum((W * d) ** 2, axis=1))
    z = theta / delta
    g = theta * norm_cdf(z) + delta * norm_pdf(z) + (tau - 1) * theta
    sigma_hat = np.mean(g)
    gamma = np.sqrt(nu**2 + d**2)

    return (
        -p
        + p * np.log(tau)
        + p * np.log(1 - tau)
        - p * np.log(sigma_hat)
        + np.sum(np.log(d))
        - np.sum(np.log(gamma))
    )


def veb_pb_jac_elbo(y, tau, nu, d, mu, W):
    p = y.shape[0]

    theta = y - mu - W @ nu
    delta = np.sqrt(np.sum((W * d) ** 2, axis=1))
    z = theta / delta

    Phi = norm_cdf(z)
    phi = norm_pdf(z)
    g = theta * Phi + delta * phi + (tau - 1) * theta
    sigma_hat = np.mean(g)
    gamma = np.sqrt(nu**2 + d**2)

    dtau = p / tau - p / (1 - tau) - np.sum(theta) / sigma_hat
    dnu = -(W.T @ ((1 - tau) - Phi)) / sigma_hat - nu / gamma / gamma
    dd = 1 / d - d / gamma / gamma - ((W**2).T @ (phi / delta)) * (d / sigma_hat)

    return dtau, dnu, dd


def veb_pb_optim_prep(
    y, tau_init, nu_init, d_init, mu, W, tau_min=1e-5, tau_max=0.25, sd_min=1e-10
):
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
    theta = y - mu - W @ nu
    delta = np.sqrt(np.sum((W * d) ** 2, axis=1))
    z = theta / delta

    Phi = norm_cdf(z)
    phi = norm_pdf(z)
    g = theta * Phi + delta * phi + (tau - 1) * theta
    sigma_hat = np.mean(g)

    return sigma_hat


def pb_map(y, mu, W, tau, sigma, gamma, mit=None, verbose=False):
    x = cp.Variable(W.shape[1])
    prob = cp.Problem(
        cp.Minimize(
            cp.sum(0.5 * cp.abs(y - mu - W @ x) + (tau - 0.5) * (y - mu - W @ x))
            + 0.5 * sigma * cp.sum_squares(x / gamma)
        )
    )

    prob.solve(solver=cp.CLARABEL, verbose=verbose)

    return mu + W @ x.value, x.value
