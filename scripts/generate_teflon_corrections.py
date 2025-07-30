import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from vebir.pca import pca
from vebir.ebs import EBS

REPO_ROOT = Path(__file__).resolve().parents[1]
PREPROC_DIR = REPO_ROOT / "data" / "preprocessed" / "teflon"
CORRECTIONS_DIR = REPO_ROOT / "data" / "corrections" / "teflon"

with open(PREPROC_DIR / "blanks_dict.pkl", "rb") as f:
    blanks_dict = pickle.load(f)

with open(PREPROC_DIR / "raw_spectra_dict.pkl", "rb") as f:
    raw_spectra_dict = pickle.load(f)

Z = np.array(blanks_dict["2011"])
ncomp_grid = 1 + np.arange(54)
tau_grid = 0.1 * np.power(2.0, np.arange(-2, 3))
a, b = np.meshgrid(ncomp_grid, tau_grid)
grid = np.column_stack([a.ravel(), b.ravel()])

print("\n===== EBS-ALS: no centering + V parameterization =====\n")

try:
    with open(CORRECTIONS_DIR / "ebs_als_dict_V.pkl", "rb") as f:
        pickle.load(f)
        print("Corrections already exist.")

except FileNotFoundError:
    mu, _, V, _ = pca(Z, detrend=False)
    mod = EBS(loss="ALS")

    ebs_als_dict = {}

    for comp in raw_spectra_dict:
        corrections_dict = {}
        Y = np.array(raw_spectra_dict[comp]["raw"])
        for j in tqdm(range(grid.shape[0]), desc=comp):
            corrections = np.zeros([Y.shape[0], Y.shape[1]])
            for i in range(corrections.shape[0]):
                mod.tau = grid[j, 1]
                mod.fit(Y[i, :], mu, V[:, : int(grid[j, 0])])
                corrections[i, :] = mod.absorbance
            corrections_dict[f"ncomp,tau = {int(grid[j, 0])},{grid[j, 1]}"] = (
                corrections
            )
        ebs_als_dict[comp] = corrections_dict

    with open(CORRECTIONS_DIR / "ebs_als_dict_V.pkl", "wb") as f:
        pickle.dump(ebs_als_dict, f)

print("\n===== EBS-ALS: centering + W parameterization =====\n")

try:
    with open(CORRECTIONS_DIR / "ebs_als_dict_W.pkl", "rb") as f:
        pickle.load(f)
        print("Corrections already exist.")

except FileNotFoundError:
    mu, _, _, W = pca(Z, detrend=True)
    mod = EBS(loss="ALS")

    ebs_als_dict = {}

    for comp in raw_spectra_dict:
        corrections_dict = {}
        Y = np.array(raw_spectra_dict[comp]["raw"])
        for j in tqdm(range(grid.shape[0]), desc=comp):
            corrections = np.zeros([Y.shape[0], Y.shape[1]])
            for i in range(corrections.shape[0]):
                mod.tau = grid[j, 1]
                mod.fit(Y[i, :], mu, W[:, : int(grid[j, 0])])
                corrections[i, :] = mod.absorbance
            corrections_dict[f"ncomp,tau = {int(grid[j, 0])},{grid[j, 1]}"] = (
                corrections
            )
        ebs_als_dict[comp] = corrections_dict

    with open(CORRECTIONS_DIR / "ebs_als_dict_W.pkl", "wb") as f:
        pickle.dump(ebs_als_dict, f)
