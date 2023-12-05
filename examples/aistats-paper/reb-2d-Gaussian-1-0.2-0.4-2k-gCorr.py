import matplotlib.pyplot as plt
plt.rc('figure', max_open_warning = 0) #no limit for fugures
plt.rcParams.update({                  #use lateX fonts
  "text.usetex": True,
  "font.family": "Helvetica"
})


# nthreads = 1

# from os import environ
# environ["OMP_NUM_THREADS"] = str(nthreads) # export OMP_NUM_THREADS=1
# environ["OPENBLAS_NUM_THREADS"] = str(nthreads) # export OPENBLAS_NUM_THREADS=1
# environ["MKL_NUM_THREADS"] = str(nthreads) # export MKL_NUM_THREADS=1
# environ["VECLIB_MAXIMUM_THREADS"] = str(nthreads) # export VECLIB_MAXIMUM_THREADS=1
# environ["NUMEXPR_NUM_THREADS"] = str(nthreads) # export NUMEXPR_NUM_THREADS=1

import numpy as np

from dadapy import *
from utils_rebuttal import run_all_methods, print_results, gauss_centered_0, cov_1_02_04, mean_0_2d

savestring="2d-Gaussian-1-0.2-0.4-2k-simple_align-1rep"
noalign=False

# import dataset
X_full = np.genfromtxt('datasets/2d-Gaussian-1-0.2-0.4-10k-X.dat')[::5]
#X_full = np.genfromtxt('datasets/2d-Gaussian-1-0.2-0.4-10k-X.dat')
np.savetxt("datasets/2d-Gaussian-1-0.2-0.4-2k-X.dat", X_full,fmt='%8f')

# generate dataset
#X_full = np.random.multivariate_normal(mean_0_2d, cov_1_02_04, 50000)

d = 2

F_full = - np.log([gauss_centered_0(x,cov_1_02_04,d) for x in X_full])
np.savetxt("datasets/2d-Gaussian-1-0.2-0.4-2k-F.dat", X_full,fmt='%8f')


print("Dataset size: ",X_full.shape[0])

nreps = 1 # number of repetitions
print("Number of repetitions: ",nreps)
nexp = 1 # number of dataset sizes

# create nreps random subsets of the 
nsample = 2000
#nsample = 1000
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
h_Scott = np.zeros((nreps, nexp))

# init error arrays
MAE_kNN_Abr = np.zeros((nreps, nexp))
MAE_kNN_Zhao = np.zeros((nreps, nexp))
MAE_kstarNN = np.zeros((nreps, nexp))
MAE_GKDE_Sil = np.zeros((nreps, nexp))
MAE_awkde = np.zeros((nreps, nexp))
MAE_GKDE_Scott = np.zeros((nreps, nexp))
MAE_PAk = np.zeros((nreps, nexp))
MAE_BMTI = np.zeros((nreps, nexp))
MAE_GMM = np.zeros((nreps, nexp))

MSE_kNN_Abr = np.zeros((nreps, nexp))
MSE_kNN_Zhao = np.zeros((nreps, nexp))
MSE_kstarNN = np.zeros((nreps, nexp))
MSE_GKDE_Sil = np.zeros((nreps, nexp))
MSE_awkde = np.zeros((nreps, nexp))
MSE_GKDE_Scott = np.zeros((nreps, nexp))
MSE_PAk = np.zeros((nreps, nexp))
MSE_BMTI = np.zeros((nreps, nexp))
MSE_GMM = np.zeros((nreps, nexp))

KLD_kNN_Abr = np.zeros((nreps, nexp))
KLD_kNN_Zhao = np.zeros((nreps, nexp))
KLD_kstarNN = np.zeros((nreps, nexp))
KLD_GKDE_Sil = np.zeros((nreps, nexp))
KLD_awkde = np.zeros((nreps, nexp))
KLD_GKDE_Scott = np.zeros((nreps, nexp))
KLD_PAk = np.zeros((nreps, nexp))
KLD_BMTI = np.zeros((nreps, nexp))
KLD_GMM = np.zeros((nreps, nexp))

# init time arrays
time_kNN_Abr = np.zeros((nreps, nexp))
time_kNN_Zhao = np.zeros((nreps, nexp))
time_kstarNN = np.zeros((nreps, nexp))
time_GKDE_Sil = np.zeros((nreps, nexp))
time_awkde = np.zeros((nreps, nexp))
time_GKDE_Scott = np.zeros((nreps, nexp))
time_PAk = np.zeros((nreps, nexp))
time_BMTI = np.zeros((nreps, nexp))
time_compute_deltaFs = np.zeros((nreps, nexp))
time_GMM = np.zeros((nreps, nexp))

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

        results = run_all_methods(X_k, F_anal_k, d=d, noalign=noalign, maxk = min(X_k.shape[0] - 1, 500))

        # assign results to arrays
        Nsample[i] = results['Nsample']
        ksel_Abr[r,i] = results['ksel_Abr']
        ksel_Zhao[r,i] = results['ksel_Zhao']
        h_Sil[r,i] = results['h_Sil']
        h_Scott[r,i] = results['h_Scott']

        MAE_kNN_Abr[r,i] = results['MAE_kNN_Abr']
        MAE_kNN_Zhao[r,i] = results['MAE_kNN_Zhao']
        MAE_kstarNN[r,i] = results['MAE_kstarNN']
        MAE_GKDE_Sil[r,i] = results['MAE_GKDE_Sil']
        MAE_awkde[r,i] = results['MAE_awkde']
        MAE_GKDE_Scott[r,i] = results['MAE_GKDE_Scott']
        MAE_PAk[r,i] = results['MAE_PAk']
        MAE_BMTI[r,i] = results['MAE_BMTI']
        MAE_GMM[r,i] = results['MAE_GMM']

        MSE_kNN_Abr[r,i] = results['MSE_kNN_Abr']
        MSE_kNN_Zhao[r,i] = results['MSE_kNN_Zhao']
        MSE_kstarNN[r,i] = results['MSE_kstarNN']
        MSE_GKDE_Sil[r,i] = results['MSE_GKDE_Sil']
        MSE_awkde[r,i] = results['MSE_awkde']
        MSE_GKDE_Scott[r,i] = results['MSE_GKDE_Scott']
        MSE_PAk[r,i] = results['MSE_PAk']
        MSE_BMTI[r,i] = results['MSE_BMTI']
        MSE_GMM[r,i] = results['MSE_GMM']
        
        if noalign:
            KLD_kNN_Abr[r,i] = results['KLD_kNN_Abr']
            KLD_kNN_Zhao[r,i] = results['KLD_kNN_Zhao']
            KLD_kstarNN[r,i] = results['KLD_kstarNN']
            KLD_GKDE_Sil[r,i] = results['KLD_GKDE_Sil']
            KLD_awkde[r,i] = results['KLD_awkde']
            KLD_GKDE_Scott[r,i] = results['KLD_GKDE_Scott']
            KLD_PAk[r,i] = results['KLD_PAk']
            KLD_BMTI[r,i] = results['KLD_BMTI']
            KLD_GMM[r,i] = results['KLD_GMM']

        time_kNN_Abr[r,i] = results['time_kNN_Abr']
        time_kNN_Zhao[r,i] = results['time_kNN_Zhao']
        time_kstarNN[r,i] = results['time_kstarNN']
        time_GKDE_Sil[r,i] = results['time_GKDE_Sil']
        time_awkde[r,i] = results['time_awkde']
        time_GKDE_Scott[r,i] = results['time_GKDE_Scott']
        time_PAk[r,i] = results['time_PAk']
        time_BMTI[r,i] = results['time_BMTI']
        time_compute_deltaFs[r,i] = results['time_compute_deltaFs']
        time_GMM[r,i] = results['time_GMM']

        if r==0:
            results_avg = results.copy()
            results_avg2 = results.copy()
            for val in results.keys():
                results_avg2[val] = results[val]**2
        else:
            for val in results.keys():
                results_avg[val] += results[val]
                results_avg2[val] += results[val]**2

        print_results(results,print_KLD=noalign)    

        if noalign:
            np.savez("results/{}.npz".format(savestring),
                Nsample=Nsample,
                ksel_Abr=ksel_Abr,
                ksel_Zhao=ksel_Zhao,
                h_Sil=h_Sil,
                h_Scott=h_Scott,

                MAE_kNN_Abr=MAE_kNN_Abr,
                MAE_kNN_Zhao=MAE_kNN_Zhao,
                MAE_kstarNN=MAE_kstarNN,
                MAE_GKDE_Sil=MAE_GKDE_Sil,
                MAE_awkde=MAE_awkde,
                MAE_GKDE_Scott=MAE_GKDE_Scott,
                MAE_PAk=MAE_PAk,
                MAE_BMTI=MAE_BMTI,
                MAE_GMM=MAE_GMM,
                
                MSE_kNN_Abr=MSE_kNN_Abr,
                MSE_kNN_Zhao=MSE_kNN_Zhao,
                MSE_kstarNN=MSE_kstarNN,
                MSE_GKDE_Sil=MSE_GKDE_Sil,
                MSE_awkde=MSE_awkde,
                MSE_GKDE_Scott=MSE_GKDE_Scott,
                MSE_PAk=MSE_PAk,
                MSE_BMTI=MSE_BMTI,
                MSE_GMM=MSE_GMM,

                KLD_kNN_Abr=KLD_kNN_Abr,
                KLD_kNN_Zhao=KLD_kNN_Zhao,
                KLD_kstarNN=KLD_kstarNN,
                KLD_GKDE_Sil=KLD_GKDE_Sil,
                KLD_awkde=KLD_awkde,
                KLD_GKDE_Scott=KLD_GKDE_Scott,
                KLD_PAk=KLD_PAk,
                KLD_BMTI=KLD_BMTI,
                KLD_GMM=KLD_GMM,
                
                time_kNN_Abr=time_kNN_Abr,
                time_kNN_Zhao=time_kNN_Zhao,
                time_kstarNN=time_kstarNN,
                time_GKDE_Sil=time_GKDE_Sil,
                time_awkde=time_awkde,
                time_GKDE_Scott=time_GKDE_Scott,
                time_PAk=time_PAk,
                time_BMTI=time_BMTI,
                time_compute_deltaFs=time_compute_deltaFs,
                time_GMM=time_GMM
            )
        else:
            np.savez("results/{}.npz".format(savestring),
                Nsample=Nsample,
                ksel_Abr=ksel_Abr,
                ksel_Zhao=ksel_Zhao,
                h_Sil=h_Sil,
                h_Scott=h_Scott,

                MAE_kNN_Abr=MAE_kNN_Abr,
                MAE_kNN_Zhao=MAE_kNN_Zhao,
                MAE_kstarNN=MAE_kstarNN,
                MAE_GKDE_Sil=MAE_GKDE_Sil,
                MAE_awkde=MAE_awkde,
                MAE_GKDE_Scott=MAE_GKDE_Scott,
                MAE_PAk=MAE_PAk,
                MAE_BMTI=MAE_BMTI,
                MAE_GMM=MAE_GMM,
                
                MSE_kNN_Abr=MSE_kNN_Abr,
                MSE_kNN_Zhao=MSE_kNN_Zhao,
                MSE_kstarNN=MSE_kstarNN,
                MSE_GKDE_Sil=MSE_GKDE_Sil,
                MSE_awkde=MSE_awkde,
                MSE_GKDE_Scott=MSE_GKDE_Scott,
                MSE_PAk=MSE_PAk,
                MSE_BMTI=MSE_BMTI,
                MSE_GMM=MSE_GMM,
                
                time_kNN_Abr=time_kNN_Abr,
                time_kNN_Zhao=time_kNN_Zhao,
                time_kstarNN=time_kstarNN,
                time_GKDE_Sil=time_GKDE_Sil,
                time_awkde=time_awkde,
                time_GKDE_Scott=time_GKDE_Scott,
                time_PAk=time_PAk,
                time_BMTI=time_BMTI,
                time_compute_deltaFs=time_compute_deltaFs,
                time_GMM=time_GMM
            )


    mean_results = results_avg.copy()
    std_results = results.copy()       #just to get the shape
    for val in results.keys():
        mean_results[val] /= nreps
        std_results[val] = (results_avg2[val]/nreps - mean_results[val]**2)/np.sqrt(nreps)

    print()
    print()
    print("AVERAGES over {} repetitions".format(nreps))
    print()
    print_results(mean_results,print_KLD=noalign)    
    print()
    print()
    print("STDEVS over {} repetitions".format(nreps))
    print()
    print_results(std_results,print_KLD=noalign)    




