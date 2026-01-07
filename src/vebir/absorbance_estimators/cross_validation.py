import numpy as np
from itertools import product
from vebir.absorbance_estimators.solvers import map_als, map_pb, pb_loss, als_loss
from tqdm import tqdm


class BlockCV:
    def __init__(
        self,
        total_wavenums,
        loss="PB",
        tau_grid=[0.1],
        c_grid=[10],
        sigma_grid=[0.0001],
        num_folds=5,
    ):
        self.loss = loss
        self.num_folds = num_folds

        self.tau_grid = tau_grid
        self.c_grid = c_grid
        self.sigma_grid = sigma_grid
        self.folds = np.array_split(np.arange(total_wavenums), num_folds)

        if self.loss == "ALS":
            self.loss_fun = als_loss
            self.comp_map = map_als
        if self.loss == "PB":
            self.loss_fun = pb_loss
            self.comp_map = map_pb

    def compute_cv_errors(self, y, mu, W, mit=100, verbose=False):
        self.cv_errs = np.zeros(
            (len(self.tau_grid), len(self.c_grid), len(self.sigma_grid))
        )
        idx_pairs = list(
            product(
                range(self.cv_errs.shape[0]),
                range(self.cv_errs.shape[1]),
                range(self.cv_errs.shape[2]),
            )
        )

        for i, j, k in tqdm(idx_pairs, disable=not verbose):
            a = np.zeros(y.shape[0])
            tau = self.tau_grid[i]
            c = self.c_grid[j]
            sigma = self.sigma_grid[k]

            for fold in self.folds:
                keep_idx = np.concatenate([f for f in self.folds if f is not fold])
                _, x = self.comp_map(
                    y[keep_idx], mu[keep_idx], W[keep_idx, :c], tau, sigma, mit
                )
                a[fold] = y[fold] - (mu[fold] + W[fold, :c] @ x)

            self.cv_errs[i, j, k] = np.sum(self.loss_fun(a, tau=tau))

        idx1, idx2, idx3 = np.unravel_index(np.argmin(self.cv_errs), self.cv_errs.shape)
        self.opt_tau = self.tau_grid[idx1]
        self.opt_c = self.c_grid[idx2]
        self.opt_sigma = self.sigma_grid[idx3]

    def estimate_absorbance(self, y, mu, W, mit=100):
        self.interference, self.x = self.comp_map(
            y, mu, W[:, : self.opt_c], self.opt_tau, self.opt_sigma, mit
        )
        self.absorbance = y - self.interference
