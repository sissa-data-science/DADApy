import matplotlib.pyplot as plt
plt.rc('figure', max_open_warning = 0) #no limit for fugures
plt.rcParams.update({                  #use lateX fonts
  "text.usetex": True,
  "font.family": "Helvetica"
})

nthreads = 1

from os import environ
environ["OMP_NUM_THREADS"] = str(nthreads) # export OMP_NUM_THREADS=1
environ["OPENBLAS_NUM_THREADS"] = str(nthreads) # export OPENBLAS_NUM_THREADS=1
environ["MKL_NUM_THREADS"] = str(nthreads) # export MKL_NUM_THREADS=1
environ["VECLIB_MAXIMUM_THREADS"] = str(nthreads) # export VECLIB_MAXIMUM_THREADS=1
environ["NUMEXPR_NUM_THREADS"] = str(nthreads) # export NUMEXPR_NUM_THREADS=1

import numpy as np

from dadapy import *
from utils_rebuttal import run_all_methods, print_results

def gaussian(x, mu, sig):
   return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))/np.sqrt(2*np.pi)

def gaussian_std(x):
   return np.exp(- np.sum(x**2) / 2)/np.sqrt((2*np.pi)**20)


def den(v):
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

#X80k = np.genfromtxt('datasets/6d_double_well-100k.txt')[40000:, 1:]

# import dataset
#X_full = np.genfromtxt('datasets/6d_double_well-1.2M.dat')[800000:, 1:] #keep 
X_full = np.genfromtxt('datasets/6d_double_well-1.2M-last_400k.txt') #keep
#print(X_full.shape) 
F_full = np.array([free(x) for x in X_full])
d = 6

print("Dataset size: ",X_full.shape[0])
# or generate dataset
# X_full = np.random.normal(0, 1, size=(500000, 20))
# F_full = np.array([free_gauss(x) for x in X_full])
# d = 20

nreps = 5 # number of repetitions
print("Number of repetitions: ",nreps)
nexp = 10 # number of dataset sizes

# create nreps random subsets of the 
nsample = 80000
print("Max batch size: ",nsample)
N = nreps*nsample #X_full.shape[0]
#print("N: ",N)
rep_indices = np.arange(N)
np.random.shuffle(rep_indices)
rep_indices = np.array_split(rep_indices, nreps)

# init arrays
Nsample = np.zeros(nexp, dtype=np.int32)
ksel_Abr = np.zeros((nreps, nexp), dtype=np.int32)
ksel_Zhao = np.zeros((nreps, nexp), dtype=np.int32)
h_Sil = np.zeros((nreps, nexp))

# init error arrays
MAE_kNN_Abr = np.zeros((nreps, nexp))
MAE_kNN_Zhao = np.zeros((nreps, nexp))
MAE_kstarNN = np.zeros((nreps, nexp))
MAE_GKDE_Sil = np.zeros((nreps, nexp))
MAE_PAk = np.zeros((nreps, nexp))
MAE_BMTI = np.zeros((nreps, nexp))
MSE_kNN_Abr = np.zeros((nreps, nexp))
MSE_kNN_Zhao = np.zeros((nreps, nexp))
MSE_kstarNN = np.zeros((nreps, nexp))
MSE_GKDE_Sil = np.zeros((nreps, nexp))
MSE_PAk = np.zeros((nreps, nexp))
MSE_BMTI = np.zeros((nreps, nexp))

# init time arrays
time_kNN_Abr = np.zeros((nreps, nexp))
time_kNN_Zhao = np.zeros((nreps, nexp))
time_kstarNN = np.zeros((nreps, nexp))
time_GKDE_Sil = np.zeros((nreps, nexp))
time_PAk = np.zeros((nreps, nexp))
time_BMTI = np.zeros((nreps, nexp))
time_compute_deltaFs = np.zeros((nreps, nexp))


# loop over dataset sizes
for i in reversed(range(0, nexp)):

    print()
    print()
    print("# -----------------------------------------------------------------")

    # loop over repetitions
    for r in range(nreps):
        Xr = X_full[rep_indices[r]]
        Fr = F_full[rep_indices[r]]
        
        X_k=Xr[::2**i]
        F_anal_k = Fr[::2**i]

        print()
        print()
        print("Batch size: ", X_k.shape[0])
        print("Repetition: ",r)
 
        results = run_all_methods(X_k, F_anal_k, d=d, kstar=10)

        # assign results to arrays
        Nsample[i] = results['Nsample']
        ksel_Abr[r,i] = results['ksel_Abr']
        ksel_Zhao[r,i] = results['ksel_Zhao']
        h_Sil[r,i] = results['h_Sil']
        MAE_kNN_Abr[r,i] = results['MAE_kNN_Abr']
        MAE_kNN_Zhao[r,i] = results['MAE_kNN_Zhao']
        MAE_kstarNN[r,i] = results['MAE_kstarNN']
        MAE_GKDE_Sil[r,i] = results['MAE_GKDE_Sil']
        MAE_PAk[r,i] = results['MAE_PAk']
        MAE_BMTI[r,i] = results['MAE_BMTI']
        MSE_kNN_Abr[r,i] = results['MSE_kNN_Abr']
        MSE_kNN_Zhao[r,i] = results['MSE_kNN_Zhao']
        MSE_kstarNN[r,i] = results['MSE_kstarNN']
        MSE_GKDE_Sil[r,i] = results['MSE_GKDE_Sil']
        MSE_PAk[r,i] = results['MSE_PAk']
        MSE_BMTI[r,i] = results['MSE_BMTI']
        time_kNN_Abr[r,i] = results['time_kNN_Abr']
        time_kNN_Zhao[r,i] = results['time_kNN_Zhao']
        time_kstarNN[r,i] = results['time_kstarNN']
        time_GKDE_Sil[r,i] = results['time_GKDE_Sil']
        time_PAk[r,i] = results['time_PAk']
        time_BMTI[r,i] = results['time_BMTI']
        time_compute_deltaFs[r,i] = results['time_compute_deltaFs']

        print_results(results)    

        # save all the above arrays to a npyz file
        np.savez("results/reb-6d-80k-gCorr.npz",
            Nsample=Nsample,
            ksel_Abr=ksel_Abr,
            ksel_Zhao=ksel_Zhao,
            h_Sil=h_Sil,
            MAE_kNN_Abr=MAE_kNN_Abr,
            MAE_kNN_Zhao=MAE_kNN_Zhao,
            MAE_kstarNN=MAE_kstarNN,
            MAE_GKDE_Sil=MAE_GKDE_Sil,
            MAE_PAk=MAE_PAk,
            MAE_BMTI=MAE_BMTI,
            MSE_kNN_Abr=MSE_kNN_Abr,
            MSE_kNN_Zhao=MSE_kNN_Zhao,
            MSE_kstarNN=MSE_kstarNN,
            MSE_GKDE_Sil=MSE_GKDE_Sil,
            MSE_PAk=MSE_PAk,
            MSE_BMTI=MSE_BMTI,
            time_kNN_Abr=time_kNN_Abr,
            time_kNN_Zhao=time_kNN_Zhao,
            time_kstarNN=time_kstarNN,
            time_GKDE_Sil=time_GKDE_Sil,
            time_PAk=time_PAk,
            time_BMTI=time_BMTI,
            time_compute_deltaFs=time_compute_deltaFs
        )