import numpy as np
import cvxpy as cp
from sklearn.linear_model import LinearRegression


def ebs_als(y, V, tau=0.1, mit=100, verbose=False):
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
    x = cp.Variable(V.shape[1])
    prob = cp.Problem(
        cp.Minimize(cp.sum(0.5 * cp.abs(y - V @ x) + (tau - 0.5) * (y - V @ x)))
    )

    prob.solve(solver=cp.CLARABEL, verbose=verbose)
    return V @ x.value, x.value


class EBS:
    """Eliminate Background Spectrum."""

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
