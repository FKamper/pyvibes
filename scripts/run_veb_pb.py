import pickle
import numpy as np
import multiprocessing as mp
import pandas as pd

from pathlib import Path
from tqdm import tqdm

from vebir.utils.metrics import compute_correlation_metrics, correlation_metrics_to_df
from vebir.interference_models.pca import pca, loo_pca
from vebir.utils.distribute import (
    distribute_veb,
    distribute_veb_c,
)

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

    with open(PREPROC_DIR / "spectra_responses.pkl", "rb") as f:
        spectra_responses = pickle.load(f)

    del ref_dict["CO2"]

    blanks_id = blanks_dict["2011"].copy().index.tolist()
    Z = np.array(blanks_dict["2011"])
    wn = np.sort(pd.read_csv(LAB_DIR / "zerofilling.txt", header=None).iloc[:, 0])

    chat_cv = loo_pca(Z)[1]
    mu, _, _, W = pca(Z)
    loss = "PB"
    print(f"OSE-LOOCV estimate: {chat_cv}")

    print("\n==================== PFTE ====================\n")
    corr_metrics = {}

    print("\n========== VEB-sigma ==========\n")
    tau_min, tau_max = 0.1, 0.1

    try:
        with open(CORRECTIONS_DIR / "sigma.pkl", "rb") as f:
            corrections_dict = pickle.load(f)
        print("Corrections exist.")

    except FileNotFoundError:
        corrections_dict = {}

        for comp in raw_spectra_dict:
            sample_id = raw_spectra_dict[comp]["raw"].copy().index.tolist()
            Y = np.array(raw_spectra_dict[comp]["raw"].copy())

            distribute_args = [
                (y, mu, W[:, :chat_cv], tau_min, tau_max, loss, sid)
                for y, sid in zip(Y, sample_id)
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

        for comp in corrections_dict:
            for entry in corrections_dict[comp]:
                entry["load"] = np.max(
                    spectra_responses[comp][["AS", "OC"]].loc[entry["sample_id"]]
                )

        for comp in ref_dict:
            gt_wn = ref_dict[comp][:, 0]
            gt_prof = ref_dict[comp][:, 1]
            for entry in corrections_dict[comp]:
                entry["ref_profile_corr"] = np.corrcoef(
                    np.interp(gt_wn, wn, entry["absorbance"]), gt_prof
                )[0, 1]

        with open(CORRECTIONS_DIR / "sigma.pkl", "wb") as f:
            pickle.dump(corrections_dict, f)

    corr_metrics["sigma"] = compute_correlation_metrics(ref_dict, corrections_dict)

    try:
        with open(CORRECTIONS_DIR / "loo_blanks_sigma.pkl", "rb") as f:
            corrections_dict = pickle.load(f)

    except:
        print("\n")

        corrections_dict = []

        for i in tqdm(
            range(Z.shape[0]),
            desc="LOO Blanks",
        ):
            y = Z[i, :]
            Zloo = np.delete(Z, i, axis=0)
            loo_chat_cv = loo_pca(Zloo)[1]
            mu_loo, _, _, W_loo = pca(Zloo)
            W_loo = W_loo[:, :loo_chat_cv]

            args = (y, mu_loo, W_loo, tau_min, tau_max, loss, blanks_id[i])
            corrections_dict.append(distribute_veb(args))

    with open(CORRECTIONS_DIR / "loo_blanks_sigma.pkl", "wb") as f:
        pickle.dump(corrections_dict, f)

    print("\n========== VEB-sigma,tau ==========\n")
    tau_min, tau_max = 1e-5, 0.5 + 1e-5

    try:
        with open(CORRECTIONS_DIR / "sigma_tau.pkl", "rb") as f:
            corrections_dict = pickle.load(f)
        print("Corrections exist.\n")

    except FileNotFoundError:
        corrections_dict = {}

        for comp in raw_spectra_dict:
            sample_id = raw_spectra_dict[comp]["raw"].copy().index.tolist()
            Y = np.array(raw_spectra_dict[comp]["raw"].copy())

            distribute_args = [
                (y, mu, W[:, :chat_cv], tau_min, tau_max, loss, sid)
                for y, sid in zip(Y, sample_id)
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

        for comp in corrections_dict:
            for entry in corrections_dict[comp]:
                entry["load"] = np.max(
                    spectra_responses[comp][["AS", "OC"]].loc[entry["sample_id"]]
                )

        for comp in ref_dict:
            gt_wn = ref_dict[comp][:, 0]
            gt_prof = ref_dict[comp][:, 1]
            for entry in corrections_dict[comp]:
                entry["ref_profile_corr"] = np.corrcoef(
                    np.interp(gt_wn, wn, entry["absorbance"]), gt_prof
                )[0, 1]

        with open(CORRECTIONS_DIR / "sigma_tau.pkl", "wb") as f:
            pickle.dump(corrections_dict, f)

    corr_metrics["sigma_tau"] = compute_correlation_metrics(ref_dict, corrections_dict)

    try:
        with open(CORRECTIONS_DIR / "loo_blanks_sigma_tau.pkl", "rb") as f:
            corrections_dict = pickle.load(f)

    except:
        corrections_dict = []

        for i in tqdm(
            range(Z.shape[0]),
            desc="LOO Blanks",
        ):
            y = Z[i, :]
            Zloo = np.delete(Z, i, axis=0)
            loo_chat_cv = loo_pca(Zloo)[1]
            mu_loo, _, _, W_loo = pca(Zloo)
            W_loo = W_loo[:, :loo_chat_cv]

            args = (y, mu_loo, W_loo, tau_min, tau_max, loss, blanks_id[i])
            corrections_dict.append(distribute_veb(args))

        with open(CORRECTIONS_DIR / "loo_blanks_sigma_tau.pkl", "wb") as f:
            pickle.dump(corrections_dict, f)

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
            sample_id = raw_spectra_dict[comp]["raw"].index.tolist()
            Y = np.array(raw_spectra_dict[comp]["raw"])

            distribute_args = [
                (y, mu, W, tau_min, tau_max, loss, c_grid, mit, sid)
                for y, sid in zip(Y, sample_id)
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

        for comp in corrections_dict:
            for entry in corrections_dict[comp]:
                entry["load"] = np.max(
                    spectra_responses[comp][["AS", "OC"]].loc[entry["sample_id"]]
                )

        for comp in ref_dict:
            gt_wn = ref_dict[comp][:, 0]
            gt_prof = ref_dict[comp][:, 1]
            for entry in corrections_dict[comp]:
                entry["ref_profile_corr"] = np.corrcoef(
                    np.interp(gt_wn, wn, entry["absorbance"]), gt_prof
                )[0, 1]

        with open(CORRECTIONS_DIR / "sigma_tau_c.pkl", "wb") as f:
            pickle.dump(corrections_dict, f)

    corr_metrics["sigma_tau_c"] = compute_correlation_metrics(
        ref_dict, corrections_dict
    )

    print("\n========== Metrics ==========\n")

    with open(CORRECTIONS_DIR / "corr_metrics.pkl", "wb") as f:
        pickle.dump(corr_metrics, f)

    print(correlation_metrics_to_df(corr_metrics))

    # print("\n==================== AIRMON ====================\n")

    # CORRECTIONS_DIR = REPO_ROOT / "data" / "corrections" / "airmon" / "veb" / "pb"
    # RAW_DIR = REPO_ROOT / "data" / "raw" / "airmon"

    # tau_min, tau_max = 1e-5, 0.25

    # names = [
    #     "ambient_measurements",
    #     "ammonium_nitrate_calibration",
    #     "ammonium_sulphate_calibration",
    #     "athens_data",
    #     "japan",
    # ]
    # corrections_dict = {}

    # for name in names:
    #     Z = pd.read_csv(RAW_DIR / f"{name}_vapor_corrected_cleaned_spectra.csv").T
    #     Y = pd.read_csv(RAW_DIR / f"{name}_vapor_corrected_loaded_spectra.csv").T
    #     wn = Z.loc["wavenumber"].values

    #     Z.drop("wavenumber", inplace=True)
    #     Y.drop("wavenumber", inplace=True)

    #     Z_na_idx = Z.isna().any(axis=0).to_numpy()
    #     Z_valid_wavenumbers_idx = np.where(~Z_na_idx)[0]
    #     Y_na_idx = Y.isna().any(axis=0).to_numpy()
    #     Y_valid_wavenumbers_idx = np.where(~Y_na_idx)[0]

    #     valid_wavenumbers_idx = np.intersect1d(
    #         Z_valid_wavenumbers_idx, Y_valid_wavenumbers_idx
    #     )

    #     Zarr = Z.iloc[:, valid_wavenumbers_idx].to_numpy()
    #     Yarr = Y.iloc[:, valid_wavenumbers_idx].to_numpy()
    #     wn = wn[valid_wavenumbers_idx]

    #     chat_cv = loo_pca(Zarr)[1]
    #     mu, _, _, W = pca(Zarr)
    #     print(f"OSE-LOOCV estimate: {chat_cv}")

    #     distribute_args = [
    #         (y, mu, W[:, :chat_cv], tau_min, tau_max, loss) for y in Yarr
    #     ]

    #     with mp.Pool(processes=mp.cpu_count()) as pool:
    #         results = list(
    #             tqdm(
    #                 pool.imap(distribute_veb, distribute_args),
    #                 total=len(distribute_args),
    #                 desc=f"{name}",
    #             )
    #         )

    #     corrections_dict[name] = results

    #     A = pd.DataFrame(np.array([i["absorbance"] for i in corrections_dict[name]]))
    #     A.columns = wn
    #     A.index = Y.index
    #     A.to_csv(CORRECTIONS_DIR / f"{name}_corrections.csv")

    #     with open(CORRECTIONS_DIR / "sigma_tau.pkl", "wb") as f:
    #         pickle.dump(corrections_dict, f)
