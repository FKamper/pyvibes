import numpy as np
import pandas as pd
import copy


# def compute_compound_cors(corrections_dict, ref_dict, wn):
#     corr_dict = {}
#     for comp in ref_dict:
#         corrections = corrections_dict[comp]
#         ref_profile = ref_dict[comp][:, 1]
#         ref_wn = ref_dict[comp][:, 0]
#         interpolated_corrections = [np.interp(ref_wn, wn, c) for c in corrections]
#         corr_dict[comp] = np.array(
#             [np.corrcoef(ref_profile, c)[0, 1] for c in interpolated_corrections]
#         )

#     return corr_dict


# def compute_method_cors(corrections_dict, ref_dict, wn):
#     corr_dict = compute_compound_cors(corrections_dict, ref_dict, wn)
#     method_cors = {}
#     all_cors = []

#     for comp in ref_dict:
#         method_cors[comp] = {
#             "compound": comp,
#             "mean": np.mean(100 * corr_dict[comp]),
#             "std_err": np.std(100 * corr_dict[comp]) / np.sqrt(len(corr_dict[comp])),
#         }
#         all_cors.append(100 * corr_dict[comp])

#     all_cors = np.concatenate(all_cors)
#     method_cors["all"] = {
#         "compound": "all",
#         "mean": np.mean(all_cors),
#         "std_err": np.std(all_cors) / np.sqrt(len(all_cors)),
#     }
#     return method_cors


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
