"""
Script to add the limit of detection (lod) estimated by the bootstrap to the corrections 
produced by run_vibes.py.

Example usage (should be run from repo root): 
python ./scripts/add_boot_lod.py -path pfte/laboratory -file PB_c_loo_tau_elbo.pkl

Notes:
1. command-line arguments are for the path to the folder containing the data and the file containing the 
   corrections for which to compute lods.
2. for each correction, the lod is estimated using B = 100 bootstrap samples and a coverage of 99%.
"""
import os
import argparse
import pickle
from vibes.inference.bootstrap import pca_bootstrap, compute_lod
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Add bootstrap limit of detection to vibes corrected spectra.")
    parser.add_argument("-path", type=str, required=True, help="Path to subfolder in data containing the spectra CSV files.")
    parser.add_argument("-file", type=str, required=True, help="Path to file containing the corrections to compute LOD for. Should be vibes")
    
    pargs = parser.parse_args()
    loss = pargs.file.split("_", 1)[0]
    base_path = f"./data/{pargs.path}"
    vibes_path = os.path.join(base_path, "vibes")
    
    with open(os.path.join(vibes_path, pargs.file), "rb") as f:
        corrections_dict = pickle.load(f)

    spectra_path = f"{base_path}/spectra.parquet"
    blank_path = f"{base_path}/blanks.parquet"

    Y = pd.read_parquet(spectra_path)
    Z = pd.read_parquet(blank_path)
    Z = np.array(Z)

    for i in tqdm(range(len(corrections_dict))):
        dct = corrections_dict[i]
        y = Y.loc[dct["sample_id"]].values
        boot_x, boot_z = pca_bootstrap(y, Z, dct["tau"], dct["sigma"], dct["c"], B=100, loss=loss, verbose=False)
        z = y - dct["absorbance"]
        xi, stdev = compute_lod(boot_z, z, alpha=0.01)

        dct["boot_x"] = boot_x
        dct["boot_z"] = boot_z
        dct["boot_stdev"] = stdev 
        dct["boot_xi"] = xi 
        dct["boot_lod"] = xi*stdev
        dct["lod_alpha"] = 0.01 
        
    with open(os.path.join(vibes_path, pargs.file), "wb") as f:
        pickle.dump(corrections_dict, f)
 
if __name__ == "__main__":
    print(f"Script started at {datetime.now()}")
    main()
    print(f"Script concluded at {datetime.now()}")