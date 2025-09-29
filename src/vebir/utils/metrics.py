import numpy as np


def compute_compound_cors(corrections_dict, ref_dict, wn):
    corr_dict = {}
    for comp in ref_dict:
        corrections = corrections_dict[comp]
        ref_profile = ref_dict[comp][:, 1]
        ref_wn = ref_dict[comp][:, 0]
        interpolated_corrections = [np.interp(ref_wn, wn, c) for c in corrections]
        corr_dict[comp] = np.array(
            [np.corrcoef(ref_profile, c)[0, 1] for c in interpolated_corrections]
        )

    return corr_dict


def compute_method_cors(corrections_dict, ref_dict, wn):
    corr_dict = compute_compound_cors(corrections_dict, ref_dict, wn)
    method_cors = {}
    all_cors = []

    for comp in ref_dict:
        method_cors[comp] = {
            "compound": comp,
            "mean": np.mean(100 * corr_dict[comp]),
            "std_err": np.std(100 * corr_dict[comp]) / np.sqrt(len(corr_dict[comp])),
        }
        all_cors.append(100 * corr_dict[comp])

    all_cors = np.concatenate(all_cors)
    method_cors["all"] = {
        "compound": "all",
        "mean": np.mean(all_cors),
        "std_err": np.std(all_cors) / np.sqrt(len(all_cors)),
    }
    return method_cors
