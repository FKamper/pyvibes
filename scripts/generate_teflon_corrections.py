import pickle
import numpy as np
import time
from pathlib import Path
from tqdm import tqdm
from vebir.pca import pca, loo_pca
from vebir.ebs import EBS
from vebir.veb import VEB, MapCV

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

print("\n===== EBS-PB: no centering + V parameterization =====\n")

try:
    with open(CORRECTIONS_DIR / "ebs_pb_dict_V.pkl", "rb") as f:
        pickle.load(f)
        print("Corrections already exist.")

except FileNotFoundError:
    mu, _, V, _ = pca(Z, detrend=False)
    mod = EBS(loss="PB")

    ebs_pb_dict = {}

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
        ebs_pb_dict[comp] = corrections_dict

    with open(CORRECTIONS_DIR / "ebs_pb_dict_V.pkl", "wb") as f:
        pickle.dump(ebs_pb_dict, f)

print("\n===== EBS-PB: centering + W parameterization =====\n")

try:
    with open(CORRECTIONS_DIR / "ebs_pb_dict_W.pkl", "rb") as f:
        pickle.load(f)
        print("Corrections already exist.")

except FileNotFoundError:
    mu, _, _, W = pca(Z, detrend=True)
    mod = EBS(loss="PB")

    ebs_pb_dict = {}

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
        ebs_pb_dict[comp] = corrections_dict

    with open(CORRECTIONS_DIR / "ebs_pb_dict_W.pkl", "wb") as f:
        pickle.dump(ebs_pb_dict, f)

print("\n===== VEB-ALS =====\n")
chat_cv = loo_pca(Z)[1]
mu, _, _, W = pca(Z, detrend=True)
W = W[:, :chat_cv]

try:
    with open(CORRECTIONS_DIR / "veb_als_dict.pkl", "rb") as f:
        pickle.load(f)
        print("Corrections already exist.")

except FileNotFoundError:
    mod = VEB(c=W.shape[1], loss="ALS")
    veb_als_dict = {}

    for comp in raw_spectra_dict:
        veb_als_dict[comp] = {}
        Y = np.array(raw_spectra_dict[comp]["raw"])
        for i in tqdm(range(Y.shape[0]), desc=comp):
            start = time.time()
            mod.fit(Y[i, :], mu, W)
            mod.map(Y[i, :], mu, W)
            end = time.time()
            veb_als_dict[comp][i] = {
                "MAP": mod.absorbance,
                "tau": mod.tau,
                "sigma_hat": mod.sigma_hat,
                "nu": mod.nu,
                "d": mod.d,
                "time": end - start,
            }

    with open(CORRECTIONS_DIR / "veb_als_dict.pkl", "wb") as f:
        pickle.dump(veb_als_dict, f)

print("\n===== VEB-ALS-Fixed =====\n")
try:
    with open(CORRECTIONS_DIR / "veb_als_dict_fixed.pkl", "rb") as f:
        pickle.load(f)
        print("Corrections already exist.")

except FileNotFoundError:
    mod = VEB(c=W.shape[1], loss="ALS")
    veb_als_dict = {}

    for comp in raw_spectra_dict:
        veb_als_dict[comp] = {}
        Y = np.array(raw_spectra_dict[comp]["raw"])
        for i in tqdm(range(Y.shape[0]), desc=comp):
            start = time.time()
            mod.fit(Y[i, :], mu, W, tau_min=0.10, tau_max=0.10)
            mod.map(Y[i, :], mu, W)
            end = time.time()
            veb_als_dict[comp][i] = {
                "MAP": mod.absorbance,
                "tau": mod.tau,
                "sigma_hat": mod.sigma_hat,
                "nu": mod.nu,
                "d": mod.d,
                "time": end - start,
            }

    with open(CORRECTIONS_DIR / "veb_als_dict_fixed.pkl", "wb") as f:
        pickle.dump(veb_als_dict, f)

print("\n===== VEB-PB =====\n")
try:
    with open(CORRECTIONS_DIR / "veb_pb_dict.pkl", "rb") as f:
        pickle.load(f)
        print("Corrections already exist.")

except FileNotFoundError:
    mod = VEB(c=W.shape[1], loss="PB")
    veb_pb_dict = {}

    for comp in raw_spectra_dict:
        veb_pb_dict[comp] = {}
        Y = np.array(raw_spectra_dict[comp]["raw"])
        for i in tqdm(range(Y.shape[0]), desc=comp):
            start = time.time()
            mod.fit(Y[i, :], mu, W)
            mod.map(Y[i, :], mu, W)
            end = time.time()
            veb_pb_dict[comp][i] = {
                "MAP": mod.absorbance,
                "tau": mod.tau,
                "sigma_hat": mod.sigma_hat,
                "nu": mod.nu,
                "d": mod.d,
                "time": end - start,
            }

    with open(CORRECTIONS_DIR / "veb_pb_dict.pkl", "wb") as f:
        pickle.dump(veb_pb_dict, f)

print("\n===== VEB-PB-Fixed =====\n")
try:
    with open(CORRECTIONS_DIR / "veb_pb_dict_fixed.pkl", "rb") as f:
        pickle.load(f)
        print("Corrections already exist.")

except FileNotFoundError:
    mod = VEB(c=W.shape[1], loss="PB")
    veb_pb_dict = {}

    for comp in raw_spectra_dict:
        veb_pb_dict[comp] = {}
        Y = np.array(raw_spectra_dict[comp]["raw"])
        for i in tqdm(range(Y.shape[0]), desc=comp):
            start = time.time()
            mod.fit(Y[i, :], mu, W, tau_min=0.10, tau_max=0.10)
            mod.map(Y[i, :], mu, W)
            end = time.time()
            veb_pb_dict[comp][i] = {
                "MAP": mod.absorbance,
                "tau": mod.tau,
                "sigma_hat": mod.sigma_hat,
                "nu": mod.nu,
                "d": mod.d,
                "time": end - start,
            }

    with open(CORRECTIONS_DIR / "veb_pb_dict_fixed.pkl", "wb") as f:
        pickle.dump(veb_pb_dict, f)

print("\n===== VEB-ALS-c =====\n")
try:
    with open(CORRECTIONS_DIR / "veb_als_dict_c.pkl", "rb") as f:
        pickle.load(f)
        print("Corrections already exist.")

except FileNotFoundError:
    mu, _, _, W = pca(Z, detrend=True)
    veb_als_dict = {}

    for comp in raw_spectra_dict:
        veb_als_dict[comp] = {}
        Y = np.array(raw_spectra_dict[comp]["raw"])
        for i in tqdm(range(Y.shape[0]), desc=comp):
            start = time.time()
            mod = VEB(c=1, loss="ALS")
            elbos = []
            for c in range(1, 54):
                mod.c = c
                mod.fit(Y[i, :], mu, W[:, :c], mit=500)
                elbos.append(mod.elbo)

                if elbos[-1] == np.max(elbos):
                    tau_init = mod.tau
                    chat = c
                    nu_init = mod.nu
                    d_init = mod.d

                mod.tau_init = mod.tau
                mod.nu_init = np.append(mod.nu_init, 0)
                mod.d_init = np.append(mod.d_init, 1)

            mod = VEB(
                c=chat, tau_init=tau_init, nu_init=nu_init, d_init=d_init, loss="ALS"
            )
            mod.fit(Y[i, :], mu, W[:, :chat])
            mod.map(Y[i, :], mu, W[:, :chat])
            end = time.time()

            veb_als_dict[comp][i] = {
                "MAP": mod.absorbance,
                "chat": chat,
                "tau": mod.tau,
                "sigma_hat": mod.sigma_hat,
                "nu": mod.nu,
                "d": mod.d,
                "time": end - start,
            }

    with open(CORRECTIONS_DIR / "veb_als_dict_c.pkl", "wb") as f:
        pickle.dump(veb_als_dict, f)

print("\n===== VEB-PB-c =====\n")
try:
    with open(CORRECTIONS_DIR / "veb_pb_dict_c.pkl", "rb") as f:
        pickle.load(f)
        print("Corrections already exist.")

except FileNotFoundError:
    mu, _, _, W = pca(Z, detrend=True)
    veb_pb_dict = {}

    for comp in raw_spectra_dict:
        veb_pb_dict[comp] = {}
        Y = np.array(raw_spectra_dict[comp]["raw"])
        for i in tqdm(range(Y.shape[0]), desc=comp):
            start = time.time()
            mod = VEB(c=1)
            elbos = []
            for c in range(1, 54):
                mod.c = c
                mod.fit(Y[i, :], mu, W[:, :c], mit=500)
                elbos.append(mod.elbo)

                if elbos[-1] == np.max(elbos):
                    tau_init = mod.tau
                    chat = c
                    nu_init = mod.nu
                    d_init = mod.d

                mod.tau_init = mod.tau
                mod.nu_init = np.append(mod.nu_init, 0)
                mod.d_init = np.append(mod.d_init, 1)

            mod = VEB(c=chat, tau_init=tau_init, nu_init=nu_init, d_init=d_init)
            mod.fit(Y[i, :], mu, W[:, :chat])
            mod.map(Y[i, :], mu, W[:, :chat])
            end = time.time()

            veb_pb_dict[comp][i] = {
                "MAP": mod.absorbance,
                "chat": chat,
                "tau": mod.tau,
                "sigma_hat": mod.sigma_hat,
                "nu": mod.nu,
                "d": mod.d,
                "time": end - start,
            }

    with open(CORRECTIONS_DIR / "veb_pb_dict_c.pkl", "wb") as f:
        pickle.dump(veb_pb_dict, f)

print("\n===== MAP-ALS-CV =====\n")
chat_cv = loo_pca(Z)[1]
mu, _, _, W = pca(Z, detrend=True)
W = W[:, :chat_cv]

try:
    with open(CORRECTIONS_DIR / "map_als_dict.pkl", "rb") as f:
        pickle.load(f)
        print("Corrections already exist.")

except FileNotFoundError:
    map_als_dict = {}

    for comp in raw_spectra_dict:
        map_als_dict[comp] = {}
        Y = np.array(raw_spectra_dict[comp]["raw"])
        for i in tqdm(range(Y.shape[0]), desc=comp):
            mod = MapCV(Y[i, :], mu, W, loss="ALS")
            start = time.time()
            mod.initialize()
            mod.search()
            mod.map()
            end = time.time()

            map_als_dict[comp][i] = {
                "MAP": mod.absorbance,
                "sigma_hat": mod.sigma,
                "time": end - start,
            }

    with open(CORRECTIONS_DIR / "map_als_dict.pkl", "wb") as f:
        pickle.dump(map_als_dict, f)

print("\n===== MAP-ALS-PB =====\n")
try:
    with open(CORRECTIONS_DIR / "map_pb_dict.pkl", "rb") as f:
        pickle.load(f)
        print("Corrections already exist.")

except FileNotFoundError:
    map_pb_dict = {}

    for comp in raw_spectra_dict:
        map_pb_dict[comp] = {}
        Y = np.array(raw_spectra_dict[comp]["raw"])
        for i in tqdm(range(Y.shape[0]), desc=comp):
            mod = MapCV(Y[i, :], mu, W, loss="PB")
            start = time.time()
            mod.initialize()
            mod.search()
            mod.map()
            end = time.time()

            map_pb_dict[comp][i] = {
                "MAP": mod.absorbance,
                "sigma_hat": mod.sigma,
                "time": end - start,
            }

    with open(CORRECTIONS_DIR / "map_pb_dict.pkl", "wb") as f:
        pickle.dump(map_pb_dict, f)
