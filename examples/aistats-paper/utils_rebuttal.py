from dadapy import *
from dadapy._utils.utils import _align_arrays
import numpy as np
import time
from awkde import GaussianKDE

def  run_all_methods(Xk, F_anal_k, d=None, kstar=None):
    # init dataset
    data = Data(Xk,verbose=False)
    data.compute_distances(maxk = min(Xk.shape[0] - 1, 100))
    if d is not None: 
        data.set_id(d)
    else:   
        data.compute_id_2NN()
    print()
    print("Nsample:")
    print(data.N)
    Nsample = data.N

    #kNN_Abr
    sec = time.perf_counter()
    ksel_Abr = Nsample**(4./(4.+d))
    data.compute_density_kNN(ksel_Abr)
    time_kNN_Abr = time.perf_counter() - sec
    off_k, F_k = _align_arrays(-data.log_den,data.log_den_err,F_anal_k)
    MAE_kNN_Abr = np.mean(np.abs(F_k-F_anal_k))
    MSE_kNN_Abr = np.mean((F_k-F_anal_k)**2)
    
    #kNN_Zhao
    sec = time.perf_counter()
    ksel_Zhao = Nsample**(2./(2.+d))
    data.compute_density_kNN(ksel_Zhao)
    time_kNN_Zhao = time.perf_counter() - sec
    off_k, F_k = _align_arrays(-data.log_den,data.log_den_err,F_anal_k)
    MAE_kNN_Zhao = np.mean(np.abs(F_k-F_anal_k))
    MSE_kNN_Zhao = np.mean((F_k-F_anal_k)**2)
    
    #kstarNN
    sec = time.perf_counter()
    data.compute_density_kstarNN()
    time_kstarNN = time.perf_counter() - sec
    off_k, F_k = _align_arrays(-data.log_den,data.log_den_err,F_anal_k)
    MAE_kstarNN = np.mean(np.abs(F_k-F_anal_k))
    MSE_kstarNN = np.mean((F_k-F_anal_k)**2)

    #GKDE Silverman
    sec = time.perf_counter()
    kdesil = GaussianKDE(glob_bw="silverman", alpha=0.0, diag_cov=True)
    kdesil.fit(data.X)
    h_Sil=kdesil.glob_bw
    F_gksil = - np.log(kdesil.predict(data.X))
    time_GKDE_Sil = time.perf_counter() - sec
    off_gksil, F_gksil = _align_arrays(F_gksil,np.ones_like(F_anal_k),F_anal_k)
    MAE_GKDE_Sil = np.mean(np.abs(F_gksil-F_anal_k))
    MSE_GKDE_Sil = np.mean((F_gksil-F_anal_k)**2)

    #PAk
    sec = time.perf_counter()
    data.compute_density_PAk()
    time_PAk = time.perf_counter() - sec
    off_k, F_k = _align_arrays(-data.log_den,data.log_den_err,F_anal_k)
    MAE_PAk = np.mean(np.abs(F_k-F_anal_k))
    MSE_PAk = np.mean((F_k-F_anal_k)**2)

    #BMTI
    
    if kstar is not None:
        data.set_kstar(kstar)
        data.compute_density_kNN(kstar)

    sec = time.perf_counter()
    data.compute_deltaFs_grads_semisum()
    time_compute_deltaFs = time.perf_counter() - sec
    sec = time.perf_counter()
    data.compute_density_gCorr(mem_efficient=False)
    time_BMTI = time.perf_counter() - sec
    off_k, F_k = _align_arrays(-data.log_den,data.log_den_err,F_anal_k)
    MAE_BMTI = np.mean(np.abs(F_k-F_anal_k))
    MSE_BMTI = np.mean((F_k-F_anal_k)**2)
    
    # define a dictionary with all the results and return it
    results = {}
    results['Nsample'] = Nsample
    results['time_kNN_Abr'] = time_kNN_Abr
    results['MAE_kNN_Abr'] = MAE_kNN_Abr
    results['MSE_kNN_Abr'] = MSE_kNN_Abr
    results['time_kNN_Zhao'] = time_kNN_Zhao
    results['MAE_kNN_Zhao'] = MAE_kNN_Zhao
    results['MSE_kNN_Zhao'] = MSE_kNN_Zhao
    results['time_kstarNN'] = time_kstarNN
    results['MAE_kstarNN'] = MAE_kstarNN
    results['MSE_kstarNN'] = MSE_kstarNN
    results['time_GKDE_Sil'] = time_GKDE_Sil
    results['MAE_GKDE_Sil'] = MAE_GKDE_Sil
    results['MSE_GKDE_Sil'] = MSE_GKDE_Sil
    results['time_PAk'] = time_PAk
    results['MAE_PAk'] = MAE_PAk
    results['MSE_PAk'] = MSE_PAk
    results['time_compute_deltaFs'] = time_compute_deltaFs
    results['time_BMTI'] = time_BMTI
    results['MAE_BMTI'] = MAE_BMTI
    results['MSE_BMTI'] = MSE_BMTI
    results['ksel_Abr'] = ksel_Abr
    results['ksel_Zhao'] = ksel_Zhao
    results['h_Sil'] = h_Sil


    return results


def print_results(results):
    print("MAE_kNN_Abr: ", results['MAE_kNN_Abr'])
    print("MAE_kNN_Zhao: ", results['MAE_kNN_Zhao'])
    print("MAE_kstarNN: ", results['MAE_kstarNN'])
    print("MAE_GKDE_Sil: ", results['MAE_GKDE_Sil'])
    print("MAE_PAk: ", results['MAE_PAk'])
    print("MAE_BMTI: ", results['MAE_BMTI'])
    print("MSE_kNN_Abr: ", results['MSE_kNN_Abr'])
    print("MSE_kNN_Zhao: ", results['MSE_kNN_Zhao'])
    print("MSE_kstarNN: ", results['MSE_kstarNN'])
    print("MSE_GKDE_Sil: ", results['MSE_GKDE_Sil'])
    print("MSE_PAk: ", results['MSE_PAk'])
    print("MSE_BMTI: ", results['MSE_BMTI'])
    print("time_kNN_Abr: ", results['time_kNN_Abr'])
    print("time_kNN_Zhao: ", results['time_kNN_Zhao'])
    print("time_kstarNN: ", results['time_kstarNN'])
    print("time_GKDE_Sil: ", results['time_GKDE_Sil'])
    print("time_PAk: ", results['time_PAk'])
    print("time_BMTI: ", results['time_BMTI'])
    print("time_compute_deltaFs: ", results['time_compute_deltaFs'])