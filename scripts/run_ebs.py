import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import pandas as pd
import numpy as np
import multiprocessing as mp
import pickle
from vibes.utils.help_functions import ebs_help_fun
from vibes.interference_models.pca import pca, loo_pca
from tqdm import tqdm
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Run EBS correction on a set of spectra stored in a parquet file.")
    parser.add_argument("-loss", type=str, required=True, help="Loss function to use when correcting the spectra.")
    parser.add_argument("-path", type=str, required=True, help="Path to subfolder in data containing the spectra CSV files.")
    parser.add_argument("-c", type=str, required=True, help="How to select the number of components for PCA. Can be loo or bcv or int. bcv = block cross-validation.")
    parser.add_argument("-tau", type=str, default="0.1", help="How to select the asymmetry parameter for the loss function. Can be a float or bcv. bcv = block cross-validation")
    parser.add_argument("-grid", type=str, default="no", help="Is the call part of a grid search? yes or no.")
    
    pargs = parser.parse_args()
    
    base_path = f"./data/{pargs.path}"
    ebs_path = os.path.join(base_path, "ebs")
    os.makedirs(ebs_path, exist_ok=True)

    if pargs.grid == "no":
        storage_path = os.path.join(ebs_path, f"{pargs.loss}_c_{pargs.c}_tau_{pargs.tau}.pkl")
    
    else:
        storage_path = os.path.join(ebs_path, f"grid/{pargs.loss}/")
        os.makedirs(storage_path, exist_ok=True)
        storage_path = os.path.join(storage_path, f"c_{pargs.c}_tau_{pargs.tau}.pkl")
        
    if os.path.isfile(storage_path):
        tqdm.write(f"Corrections corresponding {pargs.loss} loss; c={pargs.c}; tau={pargs.tau} already exist. Delete to re-run.")
        return
    
    spectra_path = f"{base_path}/spectra.parquet"
    blank_path = f"{base_path}/blanks.parquet"
    
    Y = pd.read_parquet(spectra_path)
    Z = pd.read_parquet(blank_path)
    Z = np.array(Z)
    
    mu, _, _, W = pca(Z)
    
    if pargs.c == "loo":
        c = loo_pca(Z)[1]
        print(f"Selected number of components using LOO: {c}")
    
    elif pargs.c == "bcv":
        c = None
        print("Number of components to be selected using BCV")
    
    else:
        c = int(pargs.c)

    if pargs.tau == "bcv":
        tau = None
        print("Asymmetry parameter to be selected using BCV")
    
    else:
        tau = float(pargs.tau) 

    args = [(Y.iloc[i,:].values, mu, W, tau, c, pargs.loss, Y.index[i]) for i in range(Y.shape[0])]
    
    with mp.Pool(processes=mp.cpu_count()) as pool:
        res = list(tqdm(pool.imap(ebs_help_fun, args),total=len(args),desc=f"Removing interference with: {pargs.loss} loss; c={pargs.c}; tau={pargs.tau}",leave=False,))
    
    with open(storage_path, "wb") as f:
        pickle.dump(res, f)
    
if __name__ == "__main__":
    print(f"Script started at {datetime.now()}")
    main()
    print(f"Script concluded at {datetime.now()}")



