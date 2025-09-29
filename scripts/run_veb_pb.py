import pickle
import numpy as np
import time
import multiprocessing as mp
import pandas as pd

from pathlib import Path
from tqdm import tqdm
from vebir.veb import VEB
from vebir.metrics_utils import compute_method_cors
from vebir.pca import pca, loo_pca


def distribute_function(args):
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
    }

    return res


def distribute_veb(args):
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
    }

    return res


if __name__ == "__main__":
    REPO_ROOT = Path(__file__).resolve().parents[1]
    PREPROC_DIR = REPO_ROOT / "data" / "preprocessed" / "teflon"
    CORRECTIONS_DIR = REPO_ROOT / "data" / "corrections" / "teflon" / "veb" / "pb"
    GT_DIR = REPO_ROOT / "data" / "spectrabase" / "teflon"
    LAB_DIR = REPO_ROOT / "data" / "raw" / "teflon" / "laboratory_samples"

    with open(PREPROC_DIR / "blanks_dict.pkl", "rb") as f:
        blanks_dict = pickle.load(f)

    with open(PREPROC_DIR / "raw_spectra_dict.pkl", "rb") as f:
        raw_spectra_dict = pickle.load(f)

    with open(GT_DIR / "ref_dict.pkl", "rb") as file:
        ref_dict = pickle.load(file)

    del ref_dict["CO2"]

    Z = np.array(blanks_dict["2011"])
    wn = np.sort(pd.read_csv(LAB_DIR / "zerofilling.txt", header=None).iloc[:, 0])

    chat_cv = loo_pca(Z)[1]
    mu, _, _, W = pca(Z)
    loss = "PB"
    print(f"OSE-LOOCV estimate: {chat_cv}")

    print("\n==================== PFTE ====================\n")
    df = {}

    print("\n========== VEB-sigma ==========\n")

    try:
        with open(CORRECTIONS_DIR / "sigma.pkl", "rb") as f:
            corrections_dict = pickle.load(f)
        print("Corrections exist.")

    except FileNotFoundError:
        tau_min, tau_max = 0.1, 0.1
        corrections_dict = {}

        for comp in raw_spectra_dict:
            Y = np.array(raw_spectra_dict[comp]["raw"])
            distribute_args = [
                (y, mu, W[:, :chat_cv], tau_min, tau_max, loss) for y in Y
            ]

            with mp.Pool(processes=mp.cpu_count()) as pool:
                results = list(
                    tqdm(
                        pool.imap(distribute_function, distribute_args),
                        total=len(distribute_args),
                        desc=f"{comp}",
                    )
                )

            corrections_dict[comp] = results

        with open(CORRECTIONS_DIR / "sigma.pkl", "wb") as f:
            pickle.dump(corrections_dict, f)

    for comp in ref_dict:
        corrections_dict[comp] = np.array(
            [i["absorbance"] for i in corrections_dict[comp]]
        )

    df["sigma"] = compute_method_cors(corrections_dict, ref_dict, wn)

    print("\n========== VEB-sigma,tau ==========\n")

    try:
        with open(CORRECTIONS_DIR / "sigma_tau.pkl", "rb") as f:
            corrections_dict = pickle.load(f)
        print("Corrections exist.")

    except FileNotFoundError:
        tau_min, tau_max = 1e-5, 0.25
        corrections_dict = {}

        for comp in raw_spectra_dict:
            Y = np.array(raw_spectra_dict[comp]["raw"])
            distribute_args = [
                (y, mu, W[:, :chat_cv], tau_min, tau_max, loss) for y in Y
            ]

            with mp.Pool(processes=mp.cpu_count()) as pool:
                results = list(
                    tqdm(
                        pool.imap(distribute_function, distribute_args),
                        total=len(distribute_args),
                        desc=f"{comp}",
                    )
                )

            corrections_dict[comp] = results

        with open(CORRECTIONS_DIR / "sigma_tau.pkl", "wb") as f:
            pickle.dump(corrections_dict, f)

    for comp in ref_dict:
        corrections_dict[comp] = np.array(
            [i["absorbance"] for i in corrections_dict[comp]]
        )

    df["sigma_tau"] = compute_method_cors(corrections_dict, ref_dict, wn)

    print("\n========== VEB-sigma,tau,c ==========\n")

    try:
        with open(CORRECTIONS_DIR / "sigma_tau_c.pkl", "rb") as f:
            corrections_dict = pickle.load(f)
        print("Corrections exist.")

    except FileNotFoundError:
        tau_min, tau_max = 1e-5, 0.25
        c_grid = np.arange(1, 54)
        corrections_dict = {}
        mit = 500

        for comp in raw_spectra_dict:
            Y = np.array(raw_spectra_dict[comp]["raw"])
            distribute_args = [
                (y, mu, W, tau_min, tau_max, loss, c_grid, mit) for y in Y
            ]

            with mp.Pool(processes=mp.cpu_count()) as pool:
                results = list(
                    tqdm(
                        pool.imap(distribute_veb, distribute_args),
                        total=len(distribute_args),
                        desc=f"{comp}",
                    )
                )

            corrections_dict[comp] = results

        with open(CORRECTIONS_DIR / "sigma_tau_c.pkl", "wb") as f:
            pickle.dump(corrections_dict, f)

    for comp in ref_dict:
        corrections_dict[comp] = np.array(
            [i["absorbance"] for i in corrections_dict[comp]]
        )

    df["sigma_tau_c"] = compute_method_cors(corrections_dict, ref_dict, wn)

    print("\n========== Metrics ==========\n")

    for entry, entry_dict in df.items():
        for key in entry_dict:
            entry_dict[key] = (
                f"{entry_dict[key]['mean']:.2f} pm {entry_dict[key]['std_err']:.2f}"
            )

    df = pd.DataFrame(df).T
    print(df)
    print("\n==============================================\n")
