import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from vebir.pca import loo_pca, malinowski_ind
from vebir.metrics_utils import compute_method_cors


REPO_ROOT = Path(__file__).resolve().parents[1]
CORRECTIONS_DIR = REPO_ROOT / "data" / "corrections" / "teflon"
GT_DIR = REPO_ROOT / "data" / "spectrabase" / "teflon"
LAB_DIR = REPO_ROOT / "data" / "raw" / "teflon" / "laboratory_samples"
PREPROC_DIR = REPO_ROOT / "data" / "preprocessed" / "teflon"

with open(PREPROC_DIR / "blanks_dict.pkl", "rb") as f:
    blanks_dict = pickle.load(f)

with open(GT_DIR / "ref_dict.pkl", "rb") as file:
    ref_dict = pickle.load(file)

del ref_dict["CO2"]

Z = np.array(blanks_dict["2011"])
wn = np.sort(pd.read_csv(LAB_DIR / "zerofilling.txt", header=None).iloc[:, 0])

chat_ind = malinowski_ind(Z)[0]
chat_loocv = loo_pca(Z)[1]
print(f"IND: {chat_ind}, LOOCV-OSE: {chat_loocv}")

ncomp_grid = 1 + np.arange(54)
tau_grid = 0.1 * np.power(2.0, np.arange(-2, 3))
a, b = np.meshgrid(ncomp_grid, tau_grid)
grid = np.column_stack([a.ravel(), b.ravel()])

with open(CORRECTIONS_DIR / "ebs_als_dict_V.pkl", "rb") as f:
    ebs_als_dict_V = pickle.load(f)

with open(CORRECTIONS_DIR / "ebs_als_dict_W.pkl", "rb") as f:
    ebs_als_dict_W = pickle.load(f)

with open(CORRECTIONS_DIR / "ebs_pb_dict_V.pkl", "rb") as f:
    ebs_pb_dict_V = pickle.load(f)

with open(CORRECTIONS_DIR / "ebs_pb_dict_W.pkl", "rb") as f:
    ebs_pb_dict_W = pickle.load(f)

df = {}

corrections_dict = {}
key = f"ncomp,tau = {chat_ind},{0.10}"
for comp in ref_dict:
    corrections_dict[comp] = ebs_als_dict_V[comp][key]

df["EBS-ALS-IND"] = compute_method_cors(corrections_dict, ref_dict, wn)

corrections_dict = {}
key = f"ncomp,tau = {chat_loocv},{0.10}"
for comp in ref_dict:
    corrections_dict[comp] = ebs_als_dict_W[comp][key]

df["EBS-ALS-LOOCV"] = compute_method_cors(corrections_dict, ref_dict, wn)

corrections_dict = {}
key = f"ncomp,tau = {chat_ind},{0.10}"
for comp in ref_dict:
    corrections_dict[comp] = ebs_pb_dict_V[comp][key]

df["EBS-PB-IND"] = compute_method_cors(corrections_dict, ref_dict, wn)

corrections_dict = {}
key = f"ncomp,tau = {chat_loocv},{0.10}"
for comp in ref_dict:
    corrections_dict[comp] = ebs_pb_dict_W[comp][key]

df["EBS-PB-LOOCV"] = compute_method_cors(corrections_dict, ref_dict, wn)

for entry in df:
    for key in df[entry]:
        df[entry][key] = (
            f"{df[entry][key]['mean']:.2f} pm {df[entry][key]['std_err']:.2f}"
        )

df = pd.DataFrame(df).T
print(df)
