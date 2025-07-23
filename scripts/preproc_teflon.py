import os
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw" / "teflon"
BLANKS_DIR = RAW_DIR / "blank_filters"
LAB_DIR = RAW_DIR / "laboratory_samples"
PREPROC_DIR = REPO_ROOT / "data" / "preprocessed" / "teflon"

resp = pd.read_csv(LAB_DIR / "Ruthenburg_FG_std_arealdensity.csv", skiprows=4)
wn = np.sort(pd.read_csv(LAB_DIR / "zerofilling.txt", header=None).iloc[:, 0])
compounds = np.unique(resp["compounds"])

print("\n===== Preprocessing Lab Samples =====\n")

spectra_files = os.listdir(LAB_DIR / "spectra_std")
resp_dict = {}
raw_spectra_dict = {}
spectra_responses = {}


for comp in compounds:
    resp_dict[comp] = {}
    resp_dict[comp]["response"] = resp.iloc[np.where(resp["compounds"] == comp)[0], :]


for comp in compounds:
    spectra = []
    samples = []
    resp = []

    for j in range(resp_dict[comp]["response"]["sample"].shape[0]):
        try:
            ff = spectra_files[
                np.where(
                    [
                        str(resp_dict[comp]["response"]["sample"].iloc[j]) in i
                        for i in spectra_files
                    ]
                )[0][0]
            ]
            file = LAB_DIR / "./spectra_std/" / ff
            spc = pd.read_csv(file, delimiter=" ", header=None)
            spectra.append(np.interp(wn, spc.iloc[:, 0], spc.iloc[:, 1]))
            samples.append(ff)
            resp.append(resp_dict[comp]["response"].iloc[j, 3:])
        except IndexError:
            continue

    spectra = pd.DataFrame(spectra)
    spectra.columns = wn
    resp = pd.DataFrame(resp)
    spectra.index = samples
    resp.index = samples

    raw_spectra_dict[comp] = {}
    raw_spectra_dict[comp]["raw"] = spectra
    spectra_responses[comp] = {}
    spectra_responses[comp] = resp

    print(comp + " *** Done")

# Remove outlier...this spectrum is not Suberic Acid, but something else.
raw_spectra_dict["Suberic Acid"]["raw"] = raw_spectra_dict["Suberic Acid"]["raw"].drop(
    index=raw_spectra_dict["Suberic Acid"]["raw"].index[11]
)
spectra_responses["Suberic Acid"] = spectra_responses["Suberic Acid"].drop(
    index=spectra_responses["Suberic Acid"].index[11]
)

with open(PREPROC_DIR / "raw_spectra_dict.pkl", "wb") as file:
    pickle.dump(raw_spectra_dict, file)

with open(PREPROC_DIR / "spectra_responses.pkl", "wb") as file:
    pickle.dump(spectra_responses, file)

print("\n===== Preprocessing Blanks =====\n")

blanks_dict = {}
blank_files = [
    BLANKS_DIR / "spectra_2011_blank/" / i
    for i in os.listdir(BLANKS_DIR / "spectra_2011_blank/")
]
blanks = []

for j in blank_files:
    spc = pd.read_csv(j, delimiter="\t", header=None)
    ord = np.argsort(spc.iloc[:, 0])
    spc = spc.iloc[ord, :].reset_index(drop=True)
    blanks.append(np.interp(wn, spc.iloc[:, 0], spc.iloc[:, 1]))

blanks = pd.DataFrame(blanks)
blanks.columns = wn
blanks_dict["2011"] = blanks

print("2011 *** Done")

blanks = pd.read_csv(BLANKS_DIR / "spectra_47mm_blanks.csv")
blanks47mm = []

wn47mm = np.array(blanks.iloc[:-1, 0])
ord = np.argsort(wn47mm)
wn47mm = np.sort(wn47mm)
for i in range(1, blanks.shape[1]):
    bl = np.array(blanks.iloc[:-1, i])
    blanks47mm.append(np.interp(wn, wn47mm, bl[ord]))

blanks47mm = pd.DataFrame(blanks47mm)
blanks47mm.columns = wn
blanks_dict["47mm"] = blanks47mm

print("47mm *** Done")

blank_files = [
    BLANKS_DIR / "spectra_2013_blank/" / i
    for i in os.listdir(BLANKS_DIR / "spectra_2013_blank/")
]
blanks2013 = []

for j in blank_files:
    spc = pd.read_csv(j, delimiter="\t", header=None)
    ord = np.argsort(spc.iloc[:, 0])
    spc = spc.iloc[ord, :].reset_index(drop=True)
    blanks2013.append(np.interp(wn, spc.iloc[:, 0], spc.iloc[:, 1]))

blanks2013 = pd.DataFrame(blanks2013)
blanks2013.columns = wn
blanks_dict["2013"] = blanks2013

print("2013 *** Done")


lots = os.listdir(BLANKS_DIR / "25mm/")

for k in range(len(lots)):
    lot = BLANKS_DIR / "25mm" / lots[k]
    blank_files = [lot / i for i in os.listdir(lot)]

    blanks = []

    for j in blank_files:
        spc = pd.read_csv(j, delimiter=",", header=0)
        ord = np.argsort(spc.iloc[:, 0])
        spc = spc.iloc[ord, :].reset_index(drop=True)
        blanks.append(np.interp(wn, spc.iloc[:, 0], spc.iloc[:, 1]))

    blanks = pd.DataFrame(blanks)
    blanks.columns = wn
    blanks_dict[lots[k]] = blanks

    print(f"{lots[k]} *** Done")

with open(PREPROC_DIR / "blanks_dict.pkl", "wb") as f:
    pickle.dump(blanks_dict, f)

print("Teflon Preprocessing Done")
