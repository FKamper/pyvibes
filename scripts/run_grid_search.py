import pickle
import numpy as np
import multiprocessing as mp
import pandas as pd

from pathlib import Path
from tqdm import tqdm

from vebir.interference_models.pca import pca
from vebir.utils.distribute import ref_cor_grid_search


if __name__ == "__main__":
    REPO_ROOT = Path(__file__).resolve().parents[1]
    PREPROC_DIR = REPO_ROOT / "data" / "preprocessed" / "teflon"
    CORRECTIONS_DIR = REPO_ROOT / "data" / "corrections" / "teflon"
    GT_DIR = REPO_ROOT / "data" / "spectrabase" / "teflon"
    LAB_DIR = REPO_ROOT / "data" / "raw" / "teflon" / "laboratory_samples"

    with open(PREPROC_DIR / "blanks_dict.pkl", "rb") as f:
        blanks_dict = pickle.load(f)

    with open(PREPROC_DIR / "raw_spectra_dict.pkl", "rb") as f:
        raw_spectra_dict = pickle.load(f)

    with open(GT_DIR / "ref_dict.pkl", "rb") as file:
        ref_dict = pickle.load(file)

    with open(PREPROC_DIR / "spectra_responses.pkl", "rb") as f:
        spectra_responses = pickle.load(f)

    del ref_dict["CO2"]

    blanks_id = blanks_dict["2011"].copy().index.tolist()
    Z = np.array(blanks_dict["2011"])
    wn = np.sort(pd.read_csv(LAB_DIR / "zerofilling.txt", header=None).iloc[:, 0])
    mu, _, _, W = pca(Z)

    tau_grid = 0.1 * np.power(2.0, np.arange(-2, 3))
    sigma_grid = 0.0001 * np.power(10.0, np.arange(-2, 3))
    sigma_grid = np.append(0.0, sigma_grid)
    c_grid = np.arange(1, 54, 1)

    loss = "ALS"

    print(f"\n========== {loss} ==========\n")

    try:
        with open(CORRECTIONS_DIR / f"{loss.lower()}_grid_search.pkl", "rb") as f:
            grid_search_dict = pickle.load(f)

    except:
        grid_search_dict = {}

        for comp in ref_dict:
            sample_id = raw_spectra_dict[comp]["raw"].copy().index.tolist()
            Y = np.array(raw_spectra_dict[comp]["raw"])
            ref_abs = ref_dict[comp][:, 1]
            ref_wn = ref_dict[comp][:, 0]

            distribute_args = [
                (y, mu, W, ref_wn, wn, ref_abs, tau_grid, sigma_grid, c_grid, loss, sid)
                for y, sid in zip(Y, sample_id)
            ]

            with mp.Pool(processes=mp.cpu_count()) as pool:
                results = list(
                    tqdm(
                        pool.imap(ref_cor_grid_search, distribute_args),
                        total=len(distribute_args),
                        desc=f"{comp}",
                    )
                )

            grid_search_dict[comp] = results

        with open(CORRECTIONS_DIR / f"{loss.lower()}_grid_search.pkl", "wb") as f:
            pickle.dump(grid_search_dict, f)

    loss = "PB"

    print(f"\n========== {loss} ==========\n")

    try:
        with open(CORRECTIONS_DIR / f"{loss.lower()}_grid_search.pkl", "rb") as f:
            grid_search_dict = pickle.load(f)

    except:
        grid_search_dict = {}

        for comp in ref_dict:
            sample_id = raw_spectra_dict[comp]["raw"].copy().index.tolist()
            Y = np.array(raw_spectra_dict[comp]["raw"])
            ref_abs = ref_dict[comp][:, 1]
            ref_wn = ref_dict[comp][:, 0]

            distribute_args = [
                (y, mu, W, ref_wn, wn, ref_abs, tau_grid, sigma_grid, c_grid, loss, sid)
                for y, sid in zip(Y, sample_id)
            ]

            with mp.Pool(processes=mp.cpu_count()) as pool:
                results = list(
                    tqdm(
                        pool.imap(ref_cor_grid_search, distribute_args),
                        total=len(distribute_args),
                        desc=f"{comp}",
                    )
                )

            grid_search_dict[comp] = results

        with open(CORRECTIONS_DIR / f"{loss.lower()}_grid_search.pkl", "wb") as f:
            pickle.dump(grid_search_dict, f)
