from dadapy import *
from dadapy._utils.utils import _align_arrays
import numpy as np
import time
from awkde import GaussianKDE
from sklearn.mixture import GaussianMixture

def  run_all_methods(Xk, F_anal_k, d=None, kstar=None, simple_align=False):
    # init dataset
    data = Data(Xk,verbose=False)
    data.compute_distances(maxk = min(Xk.shape[0] - 1, 100))
    
    assert d is not None, "Dimension not specified"
    data.set_id(d)
    #else:
        #data.compute_id_2NN()
    #print()
    #print("Nsample: ", data.N)
    Nsample = data.N

    #kNN_Abr
    sec = time.perf_counter()
    ksel_Abr = Nsample**(4./(4.+d))
    data.compute_density_kNN(ksel_Abr)
    time_kNN_Abr = time.perf_counter() - sec
    if simple_align is True:
        off_k, F_k = _align_arrays(-data.log_den,np.ones_like(F_anal_k),F_anal_k)
    else:    
        off_k, F_k = _align_arrays(-data.log_den,data.log_den_err,F_anal_k)
    MAE_kNN_Abr = np.mean(np.abs(F_k-F_anal_k))
    MSE_kNN_Abr = np.mean((F_k-F_anal_k)**2)
    
    #kNN_Zhao
    sec = time.perf_counter()
    ksel_Zhao = Nsample**(2./(2.+d))
    data.compute_density_kNN(ksel_Zhao)
    time_kNN_Zhao = time.perf_counter() - sec
    if simple_align is True:
        off_k, F_k = _align_arrays(-data.log_den,np.ones_like(F_anal_k),F_anal_k)
    else:    
        off_k, F_k = _align_arrays(-data.log_den,data.log_den_err,F_anal_k)
    MAE_kNN_Zhao = np.mean(np.abs(F_k-F_anal_k))
    MSE_kNN_Zhao = np.mean((F_k-F_anal_k)**2)
    
    #kstarNN
    sec = time.perf_counter()
    data.compute_density_kstarNN()
    time_kstarNN = time.perf_counter() - sec
    if simple_align is True:
        off_k, F_k = _align_arrays(-data.log_den,np.ones_like(F_anal_k),F_anal_k)
    else:    
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
    if simple_align is True:
        off_k, F_k = _align_arrays(-data.log_den,np.ones_like(F_anal_k),F_anal_k)
    else:    
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
    if simple_align is True:
        off_k, F_k = _align_arrays(-data.log_den,np.ones_like(F_anal_k),F_anal_k)
    else:    
        off_k, F_k = _align_arrays(-data.log_den,data.log_den_err,F_anal_k)
    MAE_BMTI = np.mean(np.abs(F_k-F_anal_k))
    MSE_BMTI = np.mean((F_k-F_anal_k)**2)

    # GMM
    sec = time.perf_counter()
    maxn_components = min(int(Nsample/2)+1, 20)
    gmms = np.array([GaussianMixture(n_components=n_components).fit(data.X).bic(data.X) for n_components in range(1, maxn_components)])
    best_n_components = np.argmin(gmms) + 1
    gmm = GaussianMixture(n_components=best_n_components)
    gmm.fit(data.X)
    log_dens = gmm.score_samples(data.X)
    F_predicted = -log_dens
    time_GMM = time.perf_counter() - sec
    _, F_predicted = _align_arrays(F_predicted, np.ones_like(F_anal_k), F_anal_k)
    MAE_GMM = np.mean(np.abs(F_predicted - F_anal_k))
    MSE_GMM = np.mean((F_predicted - F_anal_k) ** 2)

    
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
    results['MAE_GMM'] = MAE_GMM
    results['MSE_GMM'] = MSE_GMM
    results['time_GMM'] = time_GMM
    
    return results


def print_results(results):
    print("MAE_kNN_Abr: ", results['MAE_kNN_Abr'])
    print("MAE_kNN_Zhao: ", results['MAE_kNN_Zhao'])
    print("MAE_kstarNN: ", results['MAE_kstarNN'])
    print("MAE_GKDE_Sil: ", results['MAE_GKDE_Sil'])
    print("MAE_PAk: ", results['MAE_PAk'])
    print("MAE_BMTI: ", results['MAE_BMTI'])
    print("MAE_GMM: ", results['MAE_GMM'])
    print()
    print("MSE_kNN_Abr: ", results['MSE_kNN_Abr'])
    print("MSE_kNN_Zhao: ", results['MSE_kNN_Zhao'])
    print("MSE_kstarNN: ", results['MSE_kstarNN'])
    print("MSE_GKDE_Sil: ", results['MSE_GKDE_Sil'])
    print("MSE_PAk: ", results['MSE_PAk'])
    print("MSE_BMTI: ", results['MSE_BMTI'])
    print("MSE_GMM: ", results['MSE_GMM'])
    print()
    print("time_kNN_Abr: ", results['time_kNN_Abr'])
    print("time_kNN_Zhao: ", results['time_kNN_Zhao'])
    print("time_kstarNN: ", results['time_kstarNN'])
    print("time_GKDE_Sil: ", results['time_GKDE_Sil'])
    print("time_PAk: ", results['time_PAk'])
    print("time_BMTI: ", results['time_BMTI'])
    print("time_compute_deltaFs: ", results['time_compute_deltaFs'])
    print("time_GMM: ", results['time_GMM'])



# define some density functions
def gaussian(x, mu, sig):
   return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))/np.sqrt(2*np.pi)

def gaussian_std(x):
   return np.exp(- np.sum(x**2) / 2)/np.sqrt((2*np.pi)**20)

def den_6d(v):
    #harmonic potential = 6*(x_i)**2
    
    value = 1.    
    for i in range(2,6):
        value *= np.exp(-6*v[i]*v[i])
    
    r = value*pow( 2.*np.exp(-(-1.5 + v[0])*(-1.5 + v[0]) - (-2.5 + v[1])*(-2.5 + v[1])) + 3*np.exp(-2*v[0]*v[0] - 0.25*v[1]*v[1]) , 3 )
    #r = value*pow( 2.*np.exp(-(-1.5 + v[1])*(-1.5 + v[1]) - (-2.5 + v[0])*(-2.5 + v[0])) + 3*np.exp(-2*v[1]*v[1] - 0.25*v[0]*v[0]) , 3 )
    return r

def free(v):
    return - np.log(den(v))

def free_gauss(v):
    return - np.log(gaussian_std(v))