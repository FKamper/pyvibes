"""
Script to produce corrections for map or ebs over a grid of hyperparameter combinations.

Notes:
1. command-line arguments are for the method used to correct the spectra, the loss function and 
   the path to the folder containing the data.
2. grids are hard coded.
3. still need to add map.
"""
import itertools
import argparse
import numpy as np
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Run grid search over hyperparameter combinations for given method.")
    parser.add_argument("-method", type=str, required=True, help="Base method; either ebs, map.")
    parser.add_argument("-loss", type=str, required=True, help="Loss function to use when correcting the spectra.")
    parser.add_argument("-path", type=str, required=True, help="Path to subfolder in data containing the spectra CSV files.")
    
    pargs = parser.parse_args()

    c_grid = np.arange(1,54)
    tau_grid = np.array([0.025,0.05,0.1,0.2,0.4])
    
    if pargs.method == "ebs":
        for c, tau in itertools.product(c_grid, tau_grid):
            cmd = [
                "python", "scripts/run_ebs.py",
                "-loss", pargs.loss,
                "-path", pargs.path,
                "-c", str(c),
                "-tau", str(tau),
                "-grid", "yes"
            ]
            subprocess.run(cmd)
  
if __name__ == "__main__":
    main()