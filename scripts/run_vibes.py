import os
import argparse
import pandas as pd
import numpy as np
import multiprocessing as mp
import pickle
from vibes.interference_models.pca import pca, loo_pca
from vibes.utils.help_functions import vibes_help_fun
from tqdm import tqdm
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Run VIBES correction on a set of spectra stored in a CSV file.")
    parser.add_argument("-loss", type=str, required=True, help="Loss function to use when correcting the spectra.")
    parser.add_argument("-path", type=str, required=True, help="Path to subfolder in data containing the spectra CSV files.")
    parser.add_argument("-c", type=str, required=True, help="How to select the number of components for PCA. Can be loo or elbo.")
    parser.add_argument("-tau", type=str, default="0.1", help="How to select the asymmetry parameter for the loss function. Can be a float or elbo.")
    parser.add_argument("-mit", type=int, default=10000, help="Maximum number of iterations for the elbo optimization.")
    parser.add_argument("-num_cores", type=int, default=0, help="Number of cores to use.")

    pargs = parser.parse_args()
    
    base_path = f"./data/{pargs.path}"
    vibes_path = os.path.join(base_path, "vibes")
    os.makedirs(vibes_path, exist_ok=True)
    storage_path = os.path.join(vibes_path, f"{pargs.loss}_c_{pargs.c}_tau_{pargs.tau}.pkl")
    
    if os.path.isfile(storage_path):
        tqdm.write(f"Corrections corresponding {pargs.loss} loss; c={pargs.c}; tau={pargs.tau}; mit={pargs.mit} already exist. Delete to re-run.")
        return
    
    spectra_path = f"{base_path}/spectra.parquet"
    blank_path = f"{base_path}/blanks.parquet"
    
    Y = pd.read_parquet(spectra_path)
    Z = pd.read_parquet(blank_path)
    Z = np.array(Z)
    
    mu, _, _, W = pca(Z)
    
    if pargs.c == "loo":
        chat = loo_pca(Z)[1]
        c = chat
        print(f"Selected number of components using LOO: {chat}")
    
    else:
        chat = W.shape[1]
        c = None 
    
    W = W[:,:chat]
               
    if pargs.tau == "elbo":
        tau = None
    else:        
        tau = float(pargs.tau)   
        
    if pargs.num_cores == 0:
        num_cores = mp.cpu_count()
    else:
        num_cores = pargs.num_cores

    if num_cores == 1:
        res = []
        for i in tqdm(range(Y.shape[0]),desc=f"Removing interference with: {pargs.loss} loss; c={pargs.c}; tau={pargs.tau}; mit={pargs.mit}"):
            args = (Y.iloc[i,:].values, mu, W, tau, c, pargs.loss, pargs.mit, Y.index[i])
            res.append(vibes_help_fun(args))

    else:
        args = [(Y.iloc[i,:].values, mu, W, tau, c, pargs.loss, pargs.mit, Y.index[i]) for i in range(Y.shape[0])]
        with mp.Pool(processes = num_cores) as pool:
            res = list(tqdm(pool.imap(vibes_help_fun, args),total=len(args),desc=f"Removing interference with: {pargs.loss} loss; c={pargs.c}; tau={pargs.tau}; mit={pargs.mit}",leave=False,))
    
    with open(storage_path, "wb") as f:
        pickle.dump(res, f)
    
if __name__ == "__main__":
    print(f"Script started at {datetime.now()}")
    main()
    print(f"Script concluded at {datetime.now()}")