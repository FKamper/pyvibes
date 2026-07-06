import numpy as np
from vibes.absorbance_estimators.map import MAPEstimator
from vibes.interference_models.pca import pca
from tqdm import tqdm


def bootstrap_sample(Z):
    idx = np.random.choice(Z.shape[0], size=Z.shape[0], replace=True)
    return Z[idx]


def pca_bootstrap(y, Z, tau, sigma, c, B=100, loss="PB", verbose=False):
    boot_x = []
    boot_z = []
    map_solver = MAPEstimator(loss= loss, tau = tau, sigma = sigma)

    for b in tqdm(range(B), disable=not verbose):
        Zb = bootstrap_sample(Z)
        mub, _, _, Wb = pca(Zb)
        Wb = Wb[:, :c]
    
        z , x = map_solver.solve(y, mub, Wb)

        boot_x.append(x)
        boot_z.append(z)

    boot_x = np.array(boot_x)
    boot_z = np.array(boot_z)

    return boot_x, boot_z


def compute_lod(boot_z, z, alpha=0.01):
    stdev = np.std(boot_z, axis=0)
    t = (boot_z - z) / stdev
    xi = np.quantile(np.max(t, axis=1), 1 - alpha)
    return xi, stdev
