import numpy as np
import pandas as pd
import copy

def compute_correlation_metrics(ref_dict, corrections_dict):
    all_cors = []
    metrics = {}

    for comp in ref_dict:
        cors = np.array([entry["ref_profile_corr"] for entry in corrections_dict[comp]])
        all_cors.append(cors)

        metrics[comp] = {
            "mean": np.mean(100 * cors),
            "std_err": np.std(100 * cors) / np.sqrt(len(cors)),
        }

    all_cors = np.concatenate(all_cors)

    metrics["all"] = {
        "mean": np.mean(100 * all_cors),
        "std_err": np.std(100 * all_cors) / np.sqrt(len(all_cors)),
    }

    return metrics

def correlation_metrics_to_df(corr_metrics):
    df = copy.deepcopy(corr_metrics)

    for entry, entry_dict in df.items():
        for key in entry_dict:
            entry_dict[key] = (
                f"{entry_dict[key]['mean']:.2f} pm {entry_dict[key]['std_err']:.2f}"
            )

    df = pd.DataFrame(df).T
    return df


def compute_latent_KL_divergence(nu, d):
    return (nu**2 + d**2 - 2 * np.log(d) - 1) / 2
