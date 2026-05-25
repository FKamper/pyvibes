import time
import numpy as np
from vibes.absorbance_estimators.vibes import VibeSpec
from vibes.absorbance_estimators.cross_validation import BlockCV

def vibes_help_fun(args):
    y, mu, W, tau, c, loss, mit, sample_id = args
    
    if tau is not None:
        tau_min, tau_max = tau , tau
        tau_init = tau
        
    else:
        tau_min, tau_max = 1e-5, 0.5 + 1e-5
        tau_init = 0.1
    
    elbos_grid = []
    
    if c is None:
        nu_init = np.array([0])
        d_init = np.array([1])
          
        checkpoint = {"nu": nu_init, "d": d_init, "tau": tau_init}
        
        for i in range(W.shape[1]):
            mod = VibeSpec(c = i+1,loss=loss,nu_init=nu_init,d_init=d_init,tau_init=tau_init) 
            mod.fit(y, mu, W[:, : (i+1)], tau_min=tau_min, tau_max=tau_max,mit=500)
            elbos_grid.append(mod.elbo)
            
            if elbos_grid[-1] == np.max(elbos_grid):
                checkpoint = {"nu": mod.nu, "d": mod.d, "tau": mod.tau}
            
            nu_init = np.append(mod.nu,0)
            d_init = np.append(mod.d,1)
            tau_init = mod.tau
            
        elbos_grid = np.array(elbos_grid)
        c = np.argmax(elbos_grid) + 1        
        
        nu_init = checkpoint["nu"]
        d_init = checkpoint["d"]
        tau_init = checkpoint["tau"]
            
    else:
        nu_init = np.zeros(c)
        d_init = np.ones(c)        
            
    W = W[:, :c]
    mod = VibeSpec(c = c,loss=loss,nu_init=nu_init,d_init=d_init,tau_init=tau_init)
    start = time.time()
    mod.fit(y, mu, W, tau_min=tau_min, tau_max=tau_max, mit=mit)
    z,x = mod.map_solver.solve(y,mu,W)
    end = time.time()
    
    res = {
        "sample_id": sample_id,
        "absorbance": y-z,
        "time": end - start,
        "sigma": mod.sigma_hat,
        "tau":mod.tau,
        "c": c,
        "x": x,
        "d": mod.d,
        "nu": mod.nu,
        "elbos_grid": elbos_grid,
        "elbo": mod.elbo,
        "mit": mit,
    }

    return res

def ebs_help_fun(args):
    y, mu, W, tau, c, loss, sample_id = args
    
    sigma_grid = [0]
    
    if tau is None:
        tau_grid = 0.1 * np.power(2.0, np.arange(-2, 3))    
    
    else:
        tau_grid = [tau]
        
    if c is None:
        c_grid = np.arange(1, W.shape[1], 1)
    
    else:
        c_grid = [c]    
    
    mod = BlockCV(total_wavenums=y.shape[0], tau_grid=tau_grid,c_grid=c_grid,sigma_grid=sigma_grid,loss = loss)
    start = time.time()
    
    if tau is None or c is None:   
        mod.compute_cv_errors(y, mu, W)
    
    z,x = mod.map_solver.solve(y, mu, W[:, :mod.c], mod.mit)
    
    end = time.time()

    res = {
        "sample_id": sample_id,
        "absorbance": y-z,
        "time": end - start,
        "sigma": 0,
        "tau":mod.tau,
        "c": mod.c,
        "x": x,
    }

    return res

def map_help_fun(args):
    y, mu, W, tau, c, loss, sample_id = args
    
    if tau is None:
        tau_grid = 0.1 * np.power(2.0, np.arange(-2, 3))    
    
    else:
        tau_grid = [tau]
        
    if c is None:
        c_grid = np.arange(1, W.shape[1], 1)
    
    else:
        c_grid = [c]    
        
    sigma_grid = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    
    mod = BlockCV(total_wavenums=y.shape[0], tau_grid=tau_grid,c_grid=c_grid,sigma_grid=sigma_grid,loss = loss)
    start = time.time()
    mod.compute_cv_errors(y, mu, W)
            
    z,x = mod.map_solver.solve(y, mu, W[:, :mod.c], mod.mit)
    
    end = time.time()

    res = {
        "sample_id": sample_id,
        "absorbance": y-z,
        "time": end - start,
        "sigma": mod.sigma,
        "tau":mod.tau,
        "c": mod.c,
        "x": x,
    }

    return res

# def fun(args):
#     y, mu, W, tau, loss, sample_id = args
    
#     est = MAPEstimator(loss=loss, tau=tau, sigma=0)
#     start = time.time()
#     z,x = est.solve(y, mu, W, mit=100, verbose=False)
#     end = time.time()

#     res = {
#         "sample_id": sample_id,
#         "absorbance": y-z,
#         "time": end - start,
#         "sigma": 0,
#         "tau":tau,
#         "c": W.shape[1],
#         "x": x,
#     }

#     return res

# mod = BlockCV(total_wavenums=y.shape[0], tau_grid=tau_grid,c_grid=c_grid,sigma_grid=[sigma],)
# 
# 

# def distribute_veb(args):
#     y, mu, W, tau_min, tau_max, loss, sample_id = args

#     mod = VEB(c=W.shape[1], loss=loss)
#     start = time.time()
#     mod.fit(y, mu, W, tau_min=tau_min, tau_max=tau_max)
#     mod.map(y, mu, W)
#     end = time.time()

#     KL_components = compute_latent_KL_divergence(mod.nu, mod.d)
#     mean_KL = np.mean(KL_components)
#     mean_mdist = np.mean(mod.x**2)
#     zstats = mod.nu / mod.d

#     res = {
#         "sample_id": sample_id,
#         "absorbance": mod.absorbance,
#         "time": end - start,
#         "tau": mod.tau,
#         "c": mod.c,
#         "sigma_hat": mod.sigma_hat,
#         "nu": mod.nu,
#         "d": mod.d,
#         "x": mod.x,
#         "KL_components": KL_components,
#         "mean_KL": mean_KL,
#         "mean_mdist": mean_mdist,
#         "zstats": zstats,
#     }

#     return res


# def distribute_veb_c(args):
#     y, mu, W, tau_min, tau_max, loss, c_grid, mit, sample_id = args

#     start = time.time()
#     mod = VEB(c=c_grid[0], loss=loss)
#     elbos = []

#     for i in range(len(c_grid)):
#         c = c_grid[i]
#         mod.c = c
#         mod.fit(y, mu, W[:, :c], mit=mit, tau_min=tau_min, tau_max=tau_max)
#         elbos.append(mod.elbo)

#         if elbos[-1] == np.max(elbos):
#             tau_init = mod.tau
#             chat = c
#             nu_init = mod.nu
#             d_init = mod.d

#         if i == len(c_grid) - 1:
#             break

#         mod.tau_init = mod.tau
#         mod.nu_init = np.append(mod.nu_init, np.zeros(c_grid[i + 1] - c))
#         mod.d_init = np.append(mod.d_init, np.ones(c_grid[i + 1] - c))

#     mod = VEB(
#         c=chat,
#         tau_init=tau_init,
#         nu_init=nu_init,
#         d_init=d_init,
#         loss=loss,
#     )
#     mod.fit(
#         y,
#         mu,
#         W[:, :chat],
#         tau_min=tau_min,
#         tau_max=tau_max,
#     )
#     mod.map(y, mu, W[:, :chat])
#     end = time.time()

#     KL_components = compute_latent_KL_divergence(mod.nu, mod.d)
#     mean_KL = np.mean(KL_components)
#     mean_mdist = np.mean(mod.x**2)
#     zstats = mod.nu / mod.d

#     res = {
#         "sample_id": sample_id,
#         "absorbance": mod.absorbance,
#         "time": end - start,
#         "tau": mod.tau,
#         "c": mod.c,
#         "sigma_hat": mod.sigma_hat,
#         "nu": mod.nu,
#         "d": mod.d,
#         "x": mod.x,
#         "KL_components": KL_components,
#         "mean_KL": mean_KL,
#         "mean_mdist": mean_mdist,
#         "zstats": zstats,
#     }

#     return res


# def distribute_blockcv(args):
#     y, mu, W, lam, tau_grid, c_grid, sigma_grid, loss, sample_id = args

#     mod = BlockCV(
#         total_wavenums=y.shape[0],
#         loss=loss,
#         tau_grid=tau_grid,
#         c_grid=c_grid,
#         sigma_grid=sigma_grid,
#         num_folds=5,
#     )

#     start = time.time()
#     mod.compute_cv_errors(y, mu, W, verbose=False)
#     mod.estimate_absorbance(y, mu, W)
#     end = time.time()

#     res = {
#         "sample_id": sample_id,
#         "absorbance": mod.absorbance,
#         "time": end - start,
#         "opt_sigma": mod.opt_sigma,
#         "opt_tau": mod.opt_tau,
#         "opt_c": mod.opt_c,
#         "cv_errs": mod.cv_errs,
#         "x": mod.x,
#         "lam": lam[: mod.opt_c],
#     }

#     return res


# def ref_cor_grid_search(args):
#     y, mu, W, ref_wn, wn, ref_abs, tau_grid, sigma_grid, c_grid, loss, sid = args

#     ref_cors = np.zeros([len(tau_grid), len(sigma_grid), len(c_grid)])

#     for t in range(len(tau_grid)):
#         for s in range(len(sigma_grid)):
#             for c in range(len(c_grid)):
#                 tau = tau_grid[t]
#                 sigma = sigma_grid[s]
#                 ncomp = c_grid[c]

#                 if loss == "ALS":
#                     z, _ = map_als(
#                         y, mu, W[:, :ncomp], tau, sigma, mit=100, verbose=False
#                     )
#                 if loss == "PB":
#                     z, _ = map_pb(
#                         y, mu, W[:, :ncomp], tau, sigma, mit=100, verbose=False
#                     )
#                 a = np.interp(ref_wn, wn, y - z)

#                 ref_cors[t, s, c] = np.corrcoef(a, ref_abs)[0, 1]

#     res = {
#         "sample_id": sid,
#         "ref_cors": ref_cors,
#         "tau_grid": tau_grid,
#         "sigma_grid": sigma_grid,
#         "c_grid": c_grid,
#     }

#     return res
