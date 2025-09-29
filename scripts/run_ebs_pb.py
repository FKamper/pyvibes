import pickle
import numpy as np
import time
import multiprocessing as mp
import pandas as pd

from pathlib import Path
from tqdm import tqdm
from vebir.absorbance_estimators.ebs import EbsCV
from vebir.utils.metrics import compute_method_cors
from vebir.interference_models.pca import pca, loo_pca


def distribute_function(args):
    y, mu, W, tau_grid, c_grid, loss = args

    mod = EbsCV(y, mu, W, num_folds=5, tau_grid=tau_grid, c_grid=c_grid, loss=loss)
    start = time.time()
    mod.compute_cv_errors(verbose=False)
    mod.estimate_absorbance()
    end = time.time()

    res = {
        "absorbance": mod.absorbance,
        "time": end - start,
        "opt_tau": mod.opt_tau,
        "opt_c": mod.opt_c,
        "cv_errs": mod.cv_errs,
    }

    return res


if __name__ == "__main__":
    REPO_ROOT = Path(__file__).resolve().parents[1]
    PREPROC_DIR = REPO_ROOT / "data" / "preprocessed" / "teflon"
    CORRECTIONS_DIR = REPO_ROOT / "data" / "corrections" / "teflon" / "ebs" / "pb"
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

    print("\n========== Running Fixed ==========\n")

    try:
        with open(CORRECTIONS_DIR / "fixed.pkl", "rb") as f:
            corrections_dict = pickle.load(f)
        print("Corrections exist.")

    except FileNotFoundError:
        tau_grid = [0.1]
        c_grid = [chat_cv]

        corrections_dict = {}

        for comp in raw_spectra_dict:
            Y = np.array(raw_spectra_dict[comp]["raw"])
            distribute_args = [(y, mu, W, tau_grid, c_grid, loss) for y in Y]

            with mp.Pool(processes=mp.cpu_count()) as pool:
                results = list(
                    tqdm(
                        pool.imap(distribute_function, distribute_args),
                        total=len(distribute_args),
                        desc=f"{comp}",
                    )
                )

            corrections_dict[comp] = results

        with open(CORRECTIONS_DIR / "fixed.pkl", "wb") as f:
            pickle.dump(corrections_dict, f)

    for comp in ref_dict:
        corrections_dict[comp] = np.array(
            [i["absorbance"] for i in corrections_dict[comp]]
        )

    df["Fixed"] = compute_method_cors(corrections_dict, ref_dict, wn)

    print("\n========== Running CV-tau ==========\n")

    try:
        with open(CORRECTIONS_DIR / "cv_tau.pkl", "rb") as f:
            corrections_dict = pickle.load(f)
        print("Corrections exist.")

    except FileNotFoundError:
        tau_grid = 0.1 * np.power(2.0, np.arange(-2, 3))
        c_grid = [chat_cv]

        corrections_dict = {}

        for comp in raw_spectra_dict:
            Y = np.array(raw_spectra_dict[comp]["raw"])
            distribute_args = [(y, mu, W, tau_grid, c_grid, loss) for y in Y]

            with mp.Pool(processes=mp.cpu_count()) as pool:
                results = list(
                    tqdm(
                        pool.imap(distribute_function, distribute_args),
                        total=len(distribute_args),
                        desc=f"{comp}",
                    )
                )

            corrections_dict[comp] = results

        with open(CORRECTIONS_DIR / "cv_tau.pkl", "wb") as f:
            pickle.dump(corrections_dict, f)

    for comp in ref_dict:
        corrections_dict[comp] = np.array(
            [i["absorbance"] for i in corrections_dict[comp]]
        )

    df["CV-tau"] = compute_method_cors(corrections_dict, ref_dict, wn)

    print("\n========== Running CV-c ==========\n")

    try:
        with open(CORRECTIONS_DIR / "cv_c.pkl", "rb") as f:
            corrections_dict = pickle.load(f)
        print("Corrections exist.")

    except FileNotFoundError:
        tau_grid = [0.1]
        c_grid = np.arange(1, 53, 1)

        corrections_dict = {}

        for comp in raw_spectra_dict:
            Y = np.array(raw_spectra_dict[comp]["raw"])
            distribute_args = [(y, mu, W, tau_grid, c_grid, loss) for y in Y]

            with mp.Pool(processes=mp.cpu_count()) as pool:
                results = list(
                    tqdm(
                        pool.imap(distribute_function, distribute_args),
                        total=len(distribute_args),
                        desc=f"{comp}",
                    )
                )

            corrections_dict[comp] = results

        with open(CORRECTIONS_DIR / "cv_c.pkl", "wb") as f:
            pickle.dump(corrections_dict, f)

    for comp in ref_dict:
        corrections_dict[comp] = np.array(
            [i["absorbance"] for i in corrections_dict[comp]]
        )

    df["CV-c"] = compute_method_cors(corrections_dict, ref_dict, wn)

    print("\n========== Running CV-c-tau ==========\n")

    try:
        with open(CORRECTIONS_DIR / "cv_c_tau.pkl", "rb") as f:
            corrections_dict = pickle.load(f)
        print("Corrections exist.")

    except FileNotFoundError:
        tau_grid = 0.1 * np.power(2.0, np.arange(-2, 3))
        c_grid = np.arange(1, 53, 1)

        corrections_dict = {}

        for comp in raw_spectra_dict:
            Y = np.array(raw_spectra_dict[comp]["raw"])
            distribute_args = [(y, mu, W, tau_grid, c_grid, loss) for y in Y]

            with mp.Pool(processes=mp.cpu_count()) as pool:
                results = list(
                    tqdm(
                        pool.imap(distribute_function, distribute_args),
                        total=len(distribute_args),
                        desc=f"{comp}",
                    )
                )

            corrections_dict[comp] = results

        with open(CORRECTIONS_DIR / "cv_c_tau.pkl", "wb") as f:
            pickle.dump(corrections_dict, f)

    for comp in ref_dict:
        corrections_dict[comp] = np.array(
            [i["absorbance"] for i in corrections_dict[comp]]
        )

    df["CV-c-tau"] = compute_method_cors(corrections_dict, ref_dict, wn)

    print("\n========== Metrics ==========\n")

    for entry, entry_dict in df.items():
        for key in entry_dict:
            entry_dict[key] = (
                f"{entry_dict[key]['mean']:.2f} pm {entry_dict[key]['std_err']:.2f}"
            )

    df = pd.DataFrame(df).T
    print(df)
    print("\n==============================================\n")
