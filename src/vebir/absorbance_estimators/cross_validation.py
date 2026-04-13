import numpy as np
from itertools import product
from vebir.absorbance_estimators.map import  MAPEstimator
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
        mit=100
    ):
        self.loss = loss
        self.num_folds = num_folds
        self.mit = mit

        self.tau_grid = tau_grid
        self.c_grid = c_grid
        self.sigma_grid = sigma_grid
        self.folds = np.array_split(np.arange(total_wavenums), num_folds)
        
        self.c = c_grid[0]
        self.tau = tau_grid[0]
        self.sigma = sigma_grid[0]      
        self.map_solver = MAPEstimator(loss=self.loss, tau=self.tau, sigma=self.sigma)
        self.cv_errs = np.full(
            (len(self.tau_grid), len(self.c_grid), len(self.sigma_grid)), np.nan
        )
        
    def compute_cv_errors(self, y, mu, W, verbose=False):
        idx_pairs = list(
            product(
                range(self.cv_errs.shape[0]),
                range(self.cv_errs.shape[1]),
                range(self.cv_errs.shape[2]),
            )
        )

        for i, j, k in tqdm(idx_pairs, disable=not verbose):
            if not np.isnan(self.cv_errs[i, j, k]):
                continue
            
            a = np.zeros(y.shape[0])
            tau = self.tau_grid[i]
            c = self.c_grid[j]
            sigma = self.sigma_grid[k]

            self.map_solver.tau = tau
            self.map_solver.sigma = sigma

            for fold in self.folds:
                keep_idx = np.concatenate([f for f in self.folds if f is not fold])
                _, x = self.map_solver.solve(y[keep_idx], mu[keep_idx], W[keep_idx, :c], self.mit)
                a[fold] = y[fold] - (mu[fold] + W[fold, :c] @ x)

            self.cv_errs[i, j, k] = np.sum(self.map_solver.compute_loss(a))

        idx1, idx2, idx3 = np.unravel_index(np.argmin(self.cv_errs), self.cv_errs.shape)
        self.tau = self.tau_grid[idx1]
        self.c = self.c_grid[idx2]
        self.sigma = self.sigma_grid[idx3]
        
        self.map_solver = MAPEstimator(loss=self.loss, tau=self.tau, sigma=self.sigma)
        


    # def estimate_absorbance(self, y, mu, W, mit=100):
    #     self.interference, self.x = self.map_solver(
    #         y, mu, W[:, : self.c], self.tau, self.  sigma, mit
    #     )
    #     self.absorbance = y - self.interference

# class MapCV:
#     def __init__(
#         self,
#         y,
#         mu,
#         W,
#         loss="PB",
#         sigma_init=0.0001,
#         tau_grid=[0.1],
#         c_grid=[10],
#         num_folds=5,
#     ):
#         self.loss = loss
#         self.sigma_init = sigma_init
#         self.num_folds = num_folds
#         self.tau_grid = tau_grid
#         self.c_grid = c_grid
#         self.y = y
#         self.mu = mu
#         self.W = W
#         self.folds = np.array_split(np.arange(y.shape[0]), num_folds)

#         if self.loss == "ALS":
#             self.loss_fun = als_loss
#             self.comp_map = map_als
#         if self.loss == "PB":
#             self.loss_fun = pb_loss
#             self.comp_map = map_pb

#     def compute_cv_err(self, tau, c, sigma):
#         a = np.zeros(self.y.shape[0])
#         for fold in self.folds:
#             keep_idx = np.concatenate([f for f in self.folds if f is not fold])
#             _, x = self.comp_map(
#                 self.y[keep_idx],
#                 self.mu[keep_idx],
#                 self.W[keep_idx, :c],
#                 tau=tau,
#                 sigma=sigma,
#             )
#             a[fold] = self.y[fold] - (self.mu[fold] + self.W[fold, :c] @ x)

#         return np.sum(self.loss_fun(a, tau=tau))

#     def local_searches_sigma(self, tau, c, sigma, mit=100):
#         break_loop = False

#         prev_loss_val = self.compute_cv_err(tau, c, sigma)
#         loss_low = self.compute_cv_err(tau, c, sigma / 2)
#         loss_high = self.compute_cv_err(tau, c, sigma * 2)

#         prev_sigma = sigma
#         if prev_loss_val <= min(loss_low, loss_high):
#             break_loop = True
#             current_loss_val = prev_loss_val
#         elif loss_low < loss_high:
#             current_loss_val = loss_low
#             sigma /= 2
#         else:
#             current_loss_val = loss_high
#             sigma *= 2

#         m = 0
#         while not break_loop:
#             if sigma > prev_sigma:
#                 sigma_new = sigma * 2
#             else:
#                 sigma_new = sigma / 2

#             prev_sigma = sigma
#             prev_loss_val = current_loss_val
#             current_loss_val = self.compute_cv_err(tau, c, sigma_new)

#             if current_loss_val < prev_loss_val:
#                 sigma = sigma_new
#             else:
#                 current_loss_val = prev_loss_val
#                 break_loop = True

#             m += 1
#             if m == mit:
#                 break_loop = True

#         return sigma, current_loss_val

#     def compute_cv_errs(self, mit=100, verbose=False):
#         self.cv_errs = np.zeros((len(self.tau_grid), len(self.c_grid)))
#         self.best_sigma = np.zeros((len(self.tau_grid), len(self.c_grid)))

#         idx_pairs = list(
#             product(range(self.cv_errs.shape[0]), range(self.cv_errs.shape[1]))
#         )

#         for i, j in tqdm(idx_pairs, disable=not verbose):
#             sigma, current_loss_val = self.local_searches_sigma(
#                 self.tau_grid[i], self.c_grid[j], self.sigma_init, mit=mit
#             )

#             self.cv_errs[i, j] = current_loss_val
#             self.best_sigma[i, j] = sigma

#         row_idx, col_idx = np.unravel_index(np.argmin(self.cv_errs), self.cv_errs.shape)
#         self.opt_tau = self.tau_grid[row_idx]
#         self.opt_c = self.c_grid[col_idx]
#         self.opt_sigma = self.best_sigma[row_idx, col_idx]

#     def map(self, mit=100):
#         self.interference, self.x = self.comp_map(
#             self.y,
#             self.mu,
#             self.W[:, : self.opt_c],
#             self.opt_tau,
#             self.opt_sigma,
#             mit,
#             verbose=False,
#         )
#         self.absorbance = self.y - self.interference