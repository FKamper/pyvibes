"""
run_airmon.py

The script applied VEB to the SO4 and NO3 loaded spectra from the Airmon dataset, and performs a bootstrap analysis to estimate the limit of detection (LOD).

Note:
- some of the cleaned spectra were found to be contaminated, and are therefore excluded from the analysis. These are listed in the SO4_contaminated and NO3_contaminated lists.
- all cleaned spectra are included in the analysis excluding those from Athens, since these are still to be processed.

Author: Francois Kamper
"""

import pickle
import gc

import numpy as np
import pandas as pd
import multiprocessing as mp

from pathlib import Path
from vebir.interference_models.pca import loo_pca, pca
from vebir.utils.distribute import distribute_veb
from vebir.utils.bootstrap import pca_bootstrap, compute_lod
from tqdm import tqdm

if __name__ == "__main__":
    REPO_ROOT = Path(__file__).resolve().parents[1]
    PREPROC_DIR = REPO_ROOT / "data" / "preprocessed" / "airmon"
    CORRECTIONS_DIR = REPO_ROOT / "data" / "corrections" / "airmon"

    cleaned_spectra_dict_path = PREPROC_DIR / "cleaned_spectra_dict.pkl"

    with open(cleaned_spectra_dict_path, "rb") as f:
        cleaned_spectra_dict = pickle.load(f)

    loaded_spectra_dict_path = PREPROC_DIR / "loaded_spectra_dict.pkl"
    with open(loaded_spectra_dict_path, "rb") as f:
        loaded_spectra_dict = pickle.load(f)

    YSO4 = loaded_spectra_dict["ammonium_sulphate_calibration"].copy()
    YNO3 = loaded_spectra_dict["ammonium_nitrate_calibration"].copy()

    contaminated_names = [
        "20250603_ammoniumsulfate_calibration_3_A",
        "20250603_ammoniumsulfate_calibration_5_A",
        "20250605_ammoniumsulfate_calibration5k_3_A",
        "20250605_ammoniumsulfate_calibration5k_smallertime_5_A",
        "20250603_ammoniumsulfate_calibration_4_B",
        "20250604_ammoniumsulfate_calibration_10_B",
        "20250604_ammoniumsulfate_calibration_12_B",
        "ammonium_nitrate_calibration_5_A",
        "ammonium_nitrate_calibration_7_A",
        "ammonium_nitrate_calibration_6_B",
    ]

    cleaned_crystal_types = [
        "ammonium_sulphate_calibration",
        "ammonium_nitrate_calibration",
        "japan",
        "ambient_measurements",
    ]

    Z = pd.concat(
        [
            cleaned_spectra_dict[cleaned_crystal]
            for cleaned_crystal in cleaned_crystal_types
        ]
    )

    Z.drop(index=contaminated_names, inplace=True)
    Z = np.array(Z)

    del loaded_spectra_dict
    del cleaned_spectra_dict
    gc.collect()

    try:
        mu = np.load(CORRECTIONS_DIR / "mu.npy")
        W = np.load(CORRECTIONS_DIR / "W.npy")
        print("Loaded PCA results")

    except:
        print("Performing PCA")

        chat_min, chat_ose, Err = loo_pca(Z)
        mu, _, _, W = pca(Z)
        W = W[:, :chat_ose]

        np.save(CORRECTIONS_DIR / "mu.npy", mu)
        np.save(CORRECTIONS_DIR / "W.npy", W)

    print(f"number of components: {W.shape[1]}")

    try:
        with open(CORRECTIONS_DIR / "SO4.pkl", "rb") as f:
            SO4 = pickle.load(f)
        print("\nLoaded SO4 corrections")

    except:
        print("\nCorrecting SO4 loaded crystals")

        distribute_args = [
            (np.array(YSO4.iloc[i]), mu, W, 1e-5, 0.5 + 1e-5, "PB", YSO4.iloc[i].name)
            for i in range(len(YSO4))
        ]

        with mp.Pool(processes=mp.cpu_count()) as pool:
            SO4 = list(
                tqdm(
                    pool.imap(distribute_veb, distribute_args),
                    total=len(distribute_args),
                )
            )

        with open(CORRECTIONS_DIR / "SO4.pkl", "wb") as f:
            pickle.dump(SO4, f)

    for i in tqdm(SO4, desc="Bootstrapping SO4 corrections"):
        if "boot_z" in i:
            continue

        else:
            y = np.array(YSO4.loc[i["sample_id"]])
            z = y - np.array(i["absorbance"])
            tau = np.array(i["tau"])
            sigma = i["sigma_hat"]
            boot_x, boot_z = pca_bootstrap(
                y, Z, tau, sigma, W.shape[1], B=100, loss="PB", verbose=False
            )
            i["boot_x"] = boot_x
            i["boot_z"] = boot_z
            xi, stdev = compute_lod(boot_z, z, alpha=0.01)
            i["xi"] = xi
            i["stdev"] = stdev

        with open(CORRECTIONS_DIR / "SO4.pkl", "wb") as f:
            pickle.dump(SO4, f)

    try:
        with open(CORRECTIONS_DIR / "NO3.pkl", "rb") as f:
            NO3 = pickle.load(f)
        print("\nLoaded NO3 corrections")

    except:
        print("\nCorrecting NO3 loaded crystals")

        distribute_args = [
            (np.array(YNO3.iloc[i]), mu, W, 1e-5, 0.5 + 1e-5, "PB", YNO3.iloc[i].name)
            for i in range(len(YNO3))
        ]

        with mp.Pool(processes=mp.cpu_count()) as pool:
            NO3 = list(
                tqdm(
                    pool.imap(distribute_veb, distribute_args),
                    total=len(distribute_args),
                )
            )

        with open(CORRECTIONS_DIR / "NO3.pkl", "wb") as f:
            pickle.dump(NO3, f)

    for i in tqdm(NO3, desc="Bootstrapping NO3 corrections"):
        if "boot_z" in i:
            continue

        else:
            y = np.array(YNO3.loc[i["sample_id"]])
            z = y - np.array(i["absorbance"])
            tau = np.array(i["tau"])
            sigma = i["sigma_hat"]
            boot_x, boot_z = pca_bootstrap(
                y, Z, tau, sigma, W.shape[1], B=100, loss="PB", verbose=False
            )
            i["boot_x"] = boot_x
            i["boot_z"] = boot_z
            xi, stdev = compute_lod(boot_z, z, alpha=0.01)
            i["xi"] = xi
            i["stdev"] = stdev

        with open(CORRECTIONS_DIR / "NO3.pkl", "wb") as f:
            pickle.dump(NO3, f)
