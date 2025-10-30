import pickle
import numpy as np
import multiprocessing as mp
import pandas as pd

from pathlib import Path
from tqdm import tqdm
from vebir.utils.metrics import compute_method_cors
from vebir.interference_models.pca import pca, loo_pca
from vebir.utils.distribute import distribute_veb, distribute_veb_c

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
                        pool.imap(distribute_veb, distribute_args),
                        total=len(distribute_args),
                        desc=f"{comp}",
                    )
                )

            corrections_dict[comp] = results

        with open(CORRECTIONS_DIR / "sigma.pkl", "wb") as f:
            pickle.dump(corrections_dict, f)

    df_corrections = []
    for comp in ref_dict:
        df_comp = raw_spectra_dict[comp]["raw"]
        corrections_dict[comp] = np.array(
            [i["absorbance"] for i in corrections_dict[comp]]
        )
        df_comp.iloc[:, :] = corrections_dict[comp]
        df_corrections.append(df_comp)
    df_corrections = pd.concat(df_corrections)
    df_corrections.to_csv(CORRECTIONS_DIR / "sigma_corrections.csv")

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
                        pool.imap(distribute_veb, distribute_args),
                        total=len(distribute_args),
                        desc=f"{comp}",
                    )
                )

            corrections_dict[comp] = results

        with open(CORRECTIONS_DIR / "sigma_tau.pkl", "wb") as f:
            pickle.dump(corrections_dict, f)

    df_corrections = []
    for comp in ref_dict:
        df_comp = raw_spectra_dict[comp]["raw"]
        corrections_dict[comp] = np.array(
            [i["absorbance"] for i in corrections_dict[comp]]
        )
        df_comp.iloc[:, :] = corrections_dict[comp]
        df_corrections.append(df_comp)
    df_corrections = pd.concat(df_corrections)
    df_corrections.to_csv(CORRECTIONS_DIR / "sigma_tau_corrections.csv")

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
                        pool.imap(distribute_veb_c, distribute_args),
                        total=len(distribute_args),
                        desc=f"{comp}",
                    )
                )

            corrections_dict[comp] = results

        with open(CORRECTIONS_DIR / "sigma_tau_c.pkl", "wb") as f:
            pickle.dump(corrections_dict, f)

    df_corrections = []
    for comp in ref_dict:
        df_comp = raw_spectra_dict[comp]["raw"]
        corrections_dict[comp] = np.array(
            [i["absorbance"] for i in corrections_dict[comp]]
        )
        df_comp.iloc[:, :] = corrections_dict[comp]
        df_corrections.append(df_comp)
    df_corrections = pd.concat(df_corrections)
    df_corrections.to_csv(CORRECTIONS_DIR / "sigma_tau_c_corrections.csv")

    df["sigma_tau_c"] = compute_method_cors(corrections_dict, ref_dict, wn)

    print("\n========== Metrics ==========\n")

    for entry, entry_dict in df.items():
        for key in entry_dict:
            entry_dict[key] = (
                f"{entry_dict[key]['mean']:.2f} pm {entry_dict[key]['std_err']:.2f}"
            )

    df = pd.DataFrame(df).T
    print(df)
    # print("\n==============================================\n")

    print("\n==================== AIRMON ====================\n")

    CORRECTIONS_DIR = REPO_ROOT / "data" / "corrections" / "airmon" / "veb" / "pb"
    RAW_DIR = REPO_ROOT / "data" / "raw" / "airmon"

    tau_min, tau_max = 1e-5, 0.25

    names = [
        "ambient_measurements",
        "ammonium_nitrate_calibration",
        "ammonium_sulphate_calibration",
        "athens_data",
        "japan",
    ]
    corrections_dict = {}

    for name in names:
        Z = pd.read_csv(RAW_DIR / f"{name}_vapor_corrected_cleaned_spectra.csv").T
        Y = pd.read_csv(RAW_DIR / f"{name}_vapor_corrected_loaded_spectra.csv").T
        wn = Z.loc["wavenumber"].values

        Z.drop("wavenumber", inplace=True)
        Y.drop("wavenumber", inplace=True)

        Z_na_idx = Z.isna().any(axis=0).to_numpy()
        Z_valid_wavenumbers_idx = np.where(~Z_na_idx)[0]
        Y_na_idx = Y.isna().any(axis=0).to_numpy()
        Y_valid_wavenumbers_idx = np.where(~Y_na_idx)[0]

        valid_wavenumbers_idx = np.intersect1d(
            Z_valid_wavenumbers_idx, Y_valid_wavenumbers_idx
        )

        Zarr = Z.iloc[:, valid_wavenumbers_idx].to_numpy()
        Yarr = Y.iloc[:, valid_wavenumbers_idx].to_numpy()
        wn = wn[valid_wavenumbers_idx]

        chat_cv = loo_pca(Zarr)[1]
        mu, _, _, W = pca(Zarr)
        print(f"OSE-LOOCV estimate: {chat_cv}")

        distribute_args = [
            (y, mu, W[:, :chat_cv], tau_min, tau_max, loss) for y in Yarr
        ]

        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = list(
                tqdm(
                    pool.imap(distribute_veb, distribute_args),
                    total=len(distribute_args),
                    desc=f"{name}",
                )
            )

        corrections_dict[name] = results

        A = pd.DataFrame(np.array([i["absorbance"] for i in corrections_dict[name]]))
        A.columns = wn
        A.index = Y.index
        A.to_csv(CORRECTIONS_DIR / f"{name}_corrections.csv")

        with open(CORRECTIONS_DIR / "sigma_tau.pkl", "wb") as f:
            pickle.dump(corrections_dict, f)
