import pickle
import numpy as np
import multiprocessing as mp
import pandas as pd

from pathlib import Path
from tqdm import tqdm
from vebir.utils.distribute import distribute_blockcv
from vebir.interference_models.pca import pca, loo_pca
from vebir.utils.metrics import compute_correlation_metrics, correlation_metrics_to_df


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

    with open(PREPROC_DIR / "spectra_responses.pkl", "rb") as f:
        spectra_responses = pickle.load(f)

    del ref_dict["CO2"]

    Z = np.array(blanks_dict["2011"])
    wn = np.sort(pd.read_csv(LAB_DIR / "zerofilling.txt", header=None).iloc[:, 0])
    sigma_grid = [0.0]

    chat_cv = loo_pca(Z)[1]
    mu, lam, _, W = pca(Z)
    loss = "PB"
    print(f"OSE-LOOCV estimate: {chat_cv}")

    print("\n==================== PFTE ====================\n")
    corr_metrics = {}

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
            sample_id = raw_spectra_dict[comp]["raw"].copy().index.tolist()
            Y = np.array(raw_spectra_dict[comp]["raw"])

            distribute_args = [
                (y, mu, W, lam, tau_grid, c_grid, sigma_grid, loss, sid)
                for y, sid in zip(Y, sample_id)
            ]

            with mp.Pool(processes=mp.cpu_count()) as pool:
                results = list(
                    tqdm(
                        pool.imap(distribute_blockcv, distribute_args),
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

        with open(CORRECTIONS_DIR / "fixed.pkl", "wb") as f:
            pickle.dump(corrections_dict, f)

    corr_metrics["fixed"] = compute_correlation_metrics(ref_dict, corrections_dict)

    print("\n========== Running CV-tau ==========\n")

    try:
        with open(CORRECTIONS_DIR / "tau.pkl", "rb") as f:
            corrections_dict = pickle.load(f)
        print("Corrections exist.")

    except FileNotFoundError:
        tau_grid = 0.1 * np.power(2.0, np.arange(-2, 3))
        c_grid = [chat_cv]

        corrections_dict = {}

        for comp in raw_spectra_dict:
            sample_id = raw_spectra_dict[comp]["raw"].copy().index.tolist()
            Y = np.array(raw_spectra_dict[comp]["raw"])

            distribute_args = [
                (y, mu, W, lam, tau_grid, c_grid, sigma_grid, loss, sid)
                for y, sid in zip(Y, sample_id)
            ]

            with mp.Pool(processes=mp.cpu_count()) as pool:
                results = list(
                    tqdm(
                        pool.imap(distribute_blockcv, distribute_args),
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

        with open(CORRECTIONS_DIR / "tau.pkl", "wb") as f:
            pickle.dump(corrections_dict, f)

    corr_metrics["tau"] = compute_correlation_metrics(ref_dict, corrections_dict)

    print("\n========== Running CV-c ==========\n")

    c_grid = np.arange(1, 54, 1)

    try:
        with open(CORRECTIONS_DIR / "c.pkl", "rb") as f:
            corrections_dict = pickle.load(f)
        print("Corrections exist.")

    except FileNotFoundError:
        tau_grid = [0.1]

        corrections_dict = {}

        for comp in raw_spectra_dict:
            sample_id = raw_spectra_dict[comp]["raw"].copy().index.tolist()
            Y = np.array(raw_spectra_dict[comp]["raw"])

            distribute_args = [
                (y, mu, W, lam, tau_grid, c_grid, sigma_grid, loss, sid)
                for y, sid in zip(Y, sample_id)
            ]

            with mp.Pool(processes=mp.cpu_count()) as pool:
                results = list(
                    tqdm(
                        pool.imap(distribute_blockcv, distribute_args),
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

        with open(CORRECTIONS_DIR / "c.pkl", "wb") as f:
            pickle.dump(corrections_dict, f)

    corr_metrics["c"] = compute_correlation_metrics(ref_dict, corrections_dict)

    print("\n========== Running CV-c-tau ==========\n")

    try:
        with open(CORRECTIONS_DIR / "c_tau.pkl", "rb") as f:
            corrections_dict = pickle.load(f)
        print("Corrections exist.")

    except FileNotFoundError:
        tau_grid = 0.1 * np.power(2.0, np.arange(-2, 3))

        corrections_dict = {}

        for comp in raw_spectra_dict:
            sample_id = raw_spectra_dict[comp]["raw"].copy().index.tolist()
            Y = np.array(raw_spectra_dict[comp]["raw"])

            distribute_args = [
                (y, mu, W, lam, tau_grid, c_grid, sigma_grid, loss, sid)
                for y, sid in zip(Y, sample_id)
            ]

            with mp.Pool(processes=mp.cpu_count()) as pool:
                results = list(
                    tqdm(
                        pool.imap(distribute_blockcv, distribute_args),
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

        with open(CORRECTIONS_DIR / "c_tau.pkl", "wb") as f:
            pickle.dump(corrections_dict, f)

    corr_metrics["c_tau"] = compute_correlation_metrics(ref_dict, corrections_dict)

    print("\n========== Metrics ==========\n")

    with open(CORRECTIONS_DIR / "corr_metrics.pkl", "wb") as f:
        pickle.dump(corr_metrics, f)

    print(correlation_metrics_to_df(corr_metrics))
    print("\n==============================================\n")
