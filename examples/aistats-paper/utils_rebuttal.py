from dadapy import *
from dadapy._utils.utils import _align_arrays
import numpy as np
import time
from awkde import GaussianKDE
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity

def  run_all_methods(   Xk, F_anal_k,
                        d=None,
                        kstar=None,
                        noalign=False,
                        simple_align=True,
                        maxk=None
):
    # init dataset
    data = Data(Xk,verbose=False)
    if maxk is None:
        maxk = Xk.shape[0]-1
    data.compute_distances(maxk = maxk)
    
    assert d is not None, "Dimension not specified"
    data.set_id(d)
    #else:
        #data.compute_id_2NN()
    #print()
    #print("Nsample: ", data.N)
    Nsample = data.N

    #kNN_Abr
    sec = time.perf_counter()
    ksel_Abr = int(Nsample**(4./(4.+d)))
    data.compute_density_kNN(ksel_Abr)
    time_kNN_Abr = time.perf_counter() - sec
    F_k = -data.log_den
    if noalign is True:
        KLD_kNN_Abr = np.mean(F_k-F_anal_k)    
    else:
        if simple_align is True:
            off_k, F_k = _align_arrays(F_k,np.ones_like(F_anal_k),F_anal_k)
        else:    
            off_k, F_k = _align_arrays(F_k,data.log_den_err,F_anal_k)
    MAE_kNN_Abr = np.mean(np.abs(F_k-F_anal_k))
    MSE_kNN_Abr = np.mean((F_k-F_anal_k)**2)
    
    #kNN_Zhao
    sec = time.perf_counter()
    ksel_Zhao = int(Nsample**(2./(2.+d)))
    data.compute_density_kNN(ksel_Zhao)
    time_kNN_Zhao = time.perf_counter() - sec
    F_k = -data.log_den
    if noalign is True:
        KLD_kNN_Zhao = np.mean(F_k-F_anal_k)
    else:
        if simple_align is True:
            off_k, F_k = _align_arrays(F_k,np.ones_like(F_anal_k),F_anal_k)
        else:    
            off_k, F_k = _align_arrays(F_k,data.log_den_err,F_anal_k)
    MAE_kNN_Zhao = np.mean(np.abs(F_k-F_anal_k))
    MSE_kNN_Zhao = np.mean((F_k-F_anal_k)**2)
    
    #kstarNN
    sec = time.perf_counter()
    data.compute_density_kstarNN()
    time_kstarNN = time.perf_counter() - sec
    F_k =-data.log_den
    if noalign is True:
        KLD_kstarNN = np.mean(F_k-F_anal_k)
    if simple_align is True:
        off_k, F_k = _align_arrays(F_k,np.ones_like(F_anal_k),F_anal_k)
    else:    
        off_k, F_k = _align_arrays(F_k,data.log_den_err,F_anal_k)
    MAE_kstarNN = np.mean(np.abs(F_k-F_anal_k))
    MSE_kstarNN = np.mean((F_k-F_anal_k)**2)

    #GKDE Silverman
    sec = time.perf_counter()
    kdesil = GaussianKDE(glob_bw="silverman", alpha=0.0, diag_cov=True)
    kdesil.fit(data.X)
    h_Sil=kdesil.glob_bw
    F_k = - np.log(kdesil.predict(data.X))
    time_GKDE_Sil = time.perf_counter() - sec
    if noalign is True:
        KLD_GKDE_Sil = np.mean(F_k-F_anal_k)
    else:
        off_k, F_k = _align_arrays(F_k,np.ones_like(F_anal_k),F_anal_k)
    MAE_GKDE_Sil = np.mean(np.abs(F_k-F_anal_k))
    MSE_GKDE_Sil = np.mean((F_k-F_anal_k)**2)

    #awkde
    sec = time.perf_counter()
    awkde = GaussianKDE(glob_bw="silverman", alpha=0.5, diag_cov=True)
    awkde.fit(data.X)
    F_k = - np.log(awkde.predict(data.X))
    time_awkde = time.perf_counter() - sec
    if noalign is True:
        KLD_awkde = np.mean(F_k-F_anal_k)
    else:
        off_k, F_k = _align_arrays(F_k,np.ones_like(F_anal_k),F_anal_k)
    MAE_awkde = np.mean(np.abs(F_k-F_anal_k))
    MSE_awkde = np.mean((F_k-F_anal_k)**2)

    # GKDE Scott with scikitlearn
    sec = time.perf_counter()
    h_Scott = data.N ** (-1. / (data.dims + 4.))
    kde_scott = KernelDensity(kernel='gaussian', bandwidth=h_Scott).fit(data.X)
    F_k = - kde_scott.score_samples(data.X)
    time_GKDE_Scott = time.perf_counter() - sec
    if noalign is True:
        KLD_GKDE_Scott = np.mean(F_k-F_anal_k)
    else:
        if simple_align is True:
            off_k, F_k = _align_arrays(F_k,np.ones_like(F_anal_k),F_anal_k)
        else:    
            off_k, F_k = _align_arrays(F_k,data.log_den_err,F_anal_k)
    MAE_GKDE_Scott = np.mean(np.abs(F_k - F_anal_k))
    MSE_GKDE_Scott = np.mean((F_k - F_anal_k) ** 2)

    #PAk
    sec = time.perf_counter()
    data.compute_density_PAk()
    time_PAk = time.perf_counter() - sec
    F_k = -data.log_den
    if noalign is True:
        KLD_PAk = np.mean(F_k-F_anal_k)
    else:
        if simple_align is True:
            off_k, F_k = _align_arrays(F_k,np.ones_like(F_anal_k),F_anal_k)
        else:    
            off_k, F_k = _align_arrays(F_k,data.log_den_err,F_anal_k)
    MAE_PAk = np.mean(np.abs(F_k-F_anal_k))
    MSE_PAk = np.mean((F_k-F_anal_k)**2)

    #BMTI
    if kstar is not None:
        data.set_kstar(kstar)
        data.compute_density_kNN(kstar)
    sec = time.perf_counter()
    data.compute_deltaFs_grads_semisum()
    time_compute_deltaFs = time.perf_counter() - sec
    if noalign is True:
        sec = time.perf_counter()
        data.compute_density_kstarNN_gCorr(mem_efficient=False,alpha=0.95)
        time_BMTI = time.perf_counter() - sec
        F_k = -data.log_den
        KLD_BMTI = np.mean(F_k-F_anal_k)
    else:
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
    maxn_components = min(int(Nsample/2)+1, 10)
    gmms = np.array([GaussianMixture(n_components=n_components).fit(data.X).bic(data.X) for n_components in range(1, maxn_components)])
    best_n_components = np.argmin(gmms) + 1
    gmm = GaussianMixture(n_components=best_n_components)
    gmm.fit(data.X)
    F_k = -gmm.score_samples(data.X)
    time_GMM = time.perf_counter() - sec
    if noalign is True:
        KLD_GMM = np.mean(F_k-F_anal_k)
    else:
        if simple_align is True:
            off_k, F_k = _align_arrays(F_k,np.ones_like(F_anal_k),F_anal_k)
        else:    
            off_k, F_k = _align_arrays(F_k,data.log_den_err,F_anal_k)
    MAE_GMM = np.mean(np.abs(F_k - F_anal_k))
    MSE_GMM = np.mean((F_k - F_anal_k) ** 2)

    
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

    results['time_awkde'] = time_awkde
    results['MAE_awkde'] = MAE_awkde
    results['MSE_awkde'] = MSE_awkde
    
    results['time_GKDE_Scott'] = time_GKDE_Scott
    results['MAE_GKDE_Scott'] = MAE_GKDE_Scott
    results['MSE_GKDE_Scott'] = MSE_GKDE_Scott
    
    results['time_PAk'] = time_PAk
    results['MAE_PAk'] = MAE_PAk
    results['MSE_PAk'] = MSE_PAk
    
    results['time_compute_deltaFs'] = time_compute_deltaFs
    results['time_BMTI'] = time_BMTI
    results['MAE_BMTI'] = MAE_BMTI
    results['MSE_BMTI'] = MSE_BMTI

    results['time_GMM'] = time_GMM
    results['MAE_GMM'] = MAE_GMM
    results['MSE_GMM'] = MSE_GMM

    results['ksel_Abr'] = ksel_Abr
    results['ksel_Zhao'] = ksel_Zhao
    results['h_Sil'] = h_Sil
    results['h_Scott'] = h_Scott

    
    if noalign:
        results['KLD_kNN_Abr'] = KLD_kNN_Abr
        results['KLD_kNN_Zhao'] = KLD_kNN_Zhao
        results['KLD_kstarNN'] = KLD_kstarNN
        results['KLD_GKDE_Sil'] = KLD_GKDE_Sil
        results['KLD_awkde'] = KLD_awkde
        results['KLD_GKDE_Scott'] = KLD_GKDE_Scott
        results['KLD_PAk'] = KLD_PAk
        results['KLD_BMTI'] = KLD_BMTI
        results['KLD_GMM'] = KLD_GMM

    return results


def print_results(results,print_KLD=False):
    print("MAE_kNN_Abr: ", results['MAE_kNN_Abr'])
    print("MAE_kNN_Zhao: ", results['MAE_kNN_Zhao'])
    print("MAE_kstarNN: ", results['MAE_kstarNN'])
    print("MAE_GKDE_Sil: ", results['MAE_GKDE_Sil'])
    print("MAE_awkde: ", results['MAE_awkde'])
    print("MAE_GKDE_Scott: ", results['MAE_GKDE_Scott'])
    print("MAE_PAk: ", results['MAE_PAk'])
    print("MAE_BMTI: ", results['MAE_BMTI'])
    print("MAE_GMM: ", results['MAE_GMM'])
    print()

    print("MSE_kNN_Abr: ", results['MSE_kNN_Abr'])
    print("MSE_kNN_Zhao: ", results['MSE_kNN_Zhao'])
    print("MSE_kstarNN: ", results['MSE_kstarNN'])
    print("MSE_GKDE_Sil: ", results['MSE_GKDE_Sil'])
    print("MSE_awkde: ", results['MSE_awkde'])
    print("MSE_GKDE_Scott: ", results['MSE_GKDE_Scott'])
    print("MSE_PAk: ", results['MSE_PAk'])
    print("MSE_BMTI: ", results['MSE_BMTI'])
    print("MSE_GMM: ", results['MSE_GMM'])
    print()

    if print_KLD:
        print("KLD_kNN_Abr: ", results['KLD_kNN_Abr'])
        print("KLD_kNN_Zhao: ", results['KLD_kNN_Zhao'])
        print("KLD_kstarNN: ", results['KLD_kstarNN'])
        print("KLD_GKDE_Sil: ", results['KLD_GKDE_Sil'])
        print("KLD_awkde: ", results['KLD_awkde'])
        print("KLD_GKDE_Scott: ", results['KLD_GKDE_Scott'])
        print("KLD_PAk: ", results['KLD_PAk'])
        print("KLD_BMTI: ", results['KLD_BMTI'])
        print("KLD_GMM: ", results['KLD_GMM'])
        print()

    print("time_kNN_Abr: ", results['time_kNN_Abr'])
    print("time_kNN_Zhao: ", results['time_kNN_Zhao'])
    print("time_kstarNN: ", results['time_kstarNN'])
    print("time_GKDE_Sil: ", results['time_GKDE_Sil'])
    print("time_awkde: ", results['time_awkde'])
    print("time_GKDE_Scott: ", results['time_GKDE_Scott'])
    print("time_PAk: ", results['time_PAk'])
    print("time_BMTI: ", results['time_BMTI'])
    print("time_compute_deltaFs: ", results['time_compute_deltaFs'])
    print("time_GMM: ", results['time_GMM'])









# define some density functions and relative free energies

# 1d gaussian
def gaussian_1d(x, mu, sig):
   return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))/np.sqrt(2*np.pi)

def free_gaussian_1d(v):
    return - np.log(gaussian_1d(v))


# standard 1d Gaussian
def gaussian_std_1d(x):
   return np.exp(- np.sum(x**2) / 2)/np.sqrt((2*np.pi)**2)

def free_gaussian_std_1s(v):
    return - np.log(gaussian_std_1d(v))


#6d potential
def den_6d(v):
    #harmonic potential = 6*(x_i)**2
    
    value = 1.    
    for i in range(2,6):
        value *= np.exp(-6*v[i]*v[i])
    
    r = value*pow( 2.*np.exp(-(-1.5 + v[0])*(-1.5 + v[0]) - (-2.5 + v[1])*(-2.5 + v[1])) + 3*np.exp(-2*v[0]*v[0] - 0.25*v[1]*v[1]) , 3 )
    #r = value*pow( 2.*np.exp(-(-1.5 + v[1])*(-1.5 + v[1]) - (-2.5 + v[0])*(-2.5 + v[0])) + 3*np.exp(-2*v[1]*v[1] - 0.25*v[0]*v[0]) , 3 )
    return r/13.5735572705

def free_6d(v):
    return - np.log(den_6d(v))



# parameters for 2d Gaussians
mean_0_2d = [0, 0]
# 2d Gaussian sx=1, sy=0.2, sxy=0.4
cov_1_02_04 = np.array([[1., 0.4],
                        [0.4, 0.2]])


# dim-dimensional Normal distribution centered around the origin
def gauss_centered_0(x,cov,dim):
    
    inv = np.linalg.inv(cov)
    det = np.linalg.det(cov)

    num = np.exp(-0.5 * np.dot(x.T,np.dot(inv,x)))

    den = (2*np.pi)**(dim/2.) * det**(0.5) 
    
    return num/den

def Fgauss_centered_0(x,cov,dim):

    inv = np.linalg.inv(cov)
    det = np.linalg.det(cov)
    
    num = 0.5 * np.dot(x.T,np.dot(inv,x))
    
    dengauss = (2*np.pi)**(d/2.) * det**(0.5) 
    
    return num - np.log(dengauss)