import cvxpy as cp
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from tqdm import tqdm


def pb_loss(a, tau=0.1):
    """
    Compute the component-wise pinball loss values.

    Args:
    ----------
    a : np.ndarray
        The input array for which to compute the loss values.

    Returns
    ----------
    np.ndarray
        The computed pinball loss values.
    """
    return 0.5 * np.abs(a) + (tau - 0.5) * a


def als_loss(a, tau=0.1):
    """
    Compute the component-wise asymmetrically weighted squared loss values.

     Args:
    ----------
    a : np.ndarray
        The input array for which to compute the loss values.

    Returns
    ----------
    np.ndarray
        The computed asymmetrically weighted squared loss values.
    """
    loss = a**2
    loss[a > 0] = tau * loss[a > 0]
    loss[a < 0] = (1 - tau) * loss[a < 0]

    return loss


def map_als(y, mu, W, tau, sigma, mit=100, verbose=False):
    if sigma == 0:
        reg_mod = LinearRegression(fit_intercept=False)

    else:
        reg_mod = Ridge(fit_intercept=False, alpha=sigma / 2, solver="svd")

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

    x = reg_mod.coef_

    return z, x


def map_pb(y, mu, W, tau, sigma, mit=100, verbose=False):
    x = cp.Variable(W.shape[1])

    if sigma == 0:
        prob = cp.Problem(
            cp.Minimize(
                cp.sum(0.5 * cp.abs(y - mu - W @ x) + (tau - 0.5) * (y - mu - W @ x))
            )
        )

    else:
        prob = cp.Problem(
            cp.Minimize(
                cp.sum(0.5 * cp.abs(y - mu - W @ x) + (tau - 0.5) * (y - mu - W @ x))
                + 0.5 * sigma * cp.sum_squares(x)
            )
        )

    prob.solve(solver=cp.CLARABEL, verbose=verbose)
    z = mu + W @ x.value

    return z, x.value
