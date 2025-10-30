import time
import numpy as np
from vebir.absorbance_estimators.veb import VEB


def distribute_veb(args):
    y, mu, W, tau_min, tau_max, loss = args

    mod = VEB(c=W.shape[1], loss=loss)
    start = time.time()
    mod.fit(y, mu, W, tau_min=tau_min, tau_max=tau_max)
    mod.map(y, mu, W)
    end = time.time()

    res = {
        "absorbance": mod.absorbance,
        "time": end - start,
        "tau": mod.tau,
        "c": mod.c,
        "sigma_hat": mod.sigma_hat,
        "nu": mod.nu,
        "d": mod.d,
        "x": mod.x,
    }

    return res


def distribute_veb_c(args):
    y, mu, W, tau_min, tau_max, loss, c_grid, mit = args

    start = time.time()
    mod = VEB(c=c_grid[0], loss=loss)
    elbos = []
    for i in range(len(c_grid)):
        c = c_grid[i]
        mod.c = c
        mod.fit(y, mu, W[:, :c], mit=mit, tau_min=tau_min, tau_max=tau_max)
        elbos.append(mod.elbo)

        if elbos[-1] == np.max(elbos):
            tau_init = mod.tau
            chat = c
            nu_init = mod.nu
            d_init = mod.d

        if i == len(c_grid) - 1:
            break

        mod.tau_init = mod.tau
        mod.nu_init = np.append(mod.nu_init, np.zeros(c_grid[i + 1] - c))
        mod.d_init = np.append(mod.d_init, np.ones(c_grid[i + 1] - c))

    mod = VEB(
        c=chat,
        tau_init=tau_init,
        nu_init=nu_init,
        d_init=d_init,
        loss=loss,
    )
    mod.fit(
        y,
        mu,
        W[:, :chat],
        tau_min=tau_min,
        tau_max=tau_max,
    )
    mod.map(y, mu, W[:, :chat])
    end = time.time()

    res = {
        "absorbance": mod.absorbance,
        "time": end - start,
        "tau": mod.tau,
        "c": mod.c,
        "sigma_hat": mod.sigma_hat,
        "nu": mod.nu,
        "d": mod.d,
        "x": mod.x,
    }

    return res
