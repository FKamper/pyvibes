"""
Script to remove interference from a set of observed spectra using EBS.

Example usage (should be run from repo root): 
python ./scripts/run_ebs.py -loss PB -path pfte/laboratory -c bcv -tau bcv

Notes:
1. command-line arguments are for the loss function, path to the folder containing the data, method to select the number of components,
   method to determine the asymmetry parameter, indicator if the script is called as part of a grid search and the number of cores to use. 
2. The folder containing the data should be a subfolder in repo_root/data and contain the files spectra.parquet and blanks.parquet.
3. spectra.parquet contains the spectra from which to remove the interference as rows with the index containing identifyers. 
4. blanks.parquet should contain the interference examples as rows.
5. EBS is then deployed on the specra contained in the folder under the provided settings.
6. c = loo instructs the script to estimate the number of components used to model the interference using leave-one-out cross validation. 
   c = bcv means that the number of components are selected using blocked cross validation (bcv).
7. tau = bcv means that the asymmetry parameter is selected by blocked cross validation, otherwise it is kept fixed at a supplied value.
"""
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
    parser.add_argument("-num_cores", type=int, default=0, help="Number of cores to use.")
    
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

    if pargs.num_cores == 0:
        num_cores = mp.cpu_count()
    else:
        num_cores = pargs.num_cores

    if num_cores == 1:
        res = []
        for i in tqdm(range(Y.shape[0]),desc=f"Removing interference with: {pargs.loss} loss; c={pargs.c}; tau={pargs.tau}"):
            args = (Y.iloc[i,:].values, mu, W, tau, c, pargs.loss, Y.index[i])
            res.append(ebs_help_fun(args))

    else:
        args = [(Y.iloc[i,:].values, mu, W, tau, c, pargs.loss, Y.index[i]) for i in range(Y.shape[0])]
        with mp.Pool(processes = num_cores) as pool:
            res = list(tqdm(pool.imap(ebs_help_fun, args),total=len(args),desc=f"Removing interference with: {pargs.loss} loss; c={pargs.c}; tau={pargs.tau}",leave=False,))
    
    with open(storage_path, "wb") as f:
        pickle.dump(res, f)
    
if __name__ == "__main__":
    print(f"Script started at {datetime.now()}")
    main()
    print(f"Script concluded at {datetime.now()}")



