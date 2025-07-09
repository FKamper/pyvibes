import numpy as np
import cvxpy as cp
from sklearn.linear_model import LinearRegression


def ebs_als(y, V, tau=0.1, maxit=100, verbose=False):
    reg_mod = LinearRegression(fit_intercept=False)
    w = np.repeat(tau, y.shape[0])

    for i in range(maxit):
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
