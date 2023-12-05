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

from utils_rebuttal import run_all_methods, print_results, compute_roughness

savestring="roughness-2d-MBx0.0325-5k-simple_align"
noalign=False

# import dataset
X_full = np.genfromtxt('datasets/2d-MuellerBrown_times_0.0325-5k.dat')


d = 2

F_full = np.genfromtxt('datasets/2d-MuellerBrown_times_0.0325-F_anal-5k.dat')


print("Dataset size: ",X_full.shape[0])

nreps = 1 # number of repetitions
print("Number of repetitions: ",nreps)
nexp = 1 # number of dataset sizes

# create nreps random subsets of the 
nsample = 5000
#nsample = 1000
print("Max batch size: ",nsample)
N = nreps*nsample #X_full.shape[0]
#print("N: ",N)
rep_indices = np.arange(N)
np.random.shuffle(rep_indices)
rep_indices = np.array_split(rep_indices, nreps)

# init arrays
Nsample = np.zeros(nexp, dtype=np.int32)

# init error arrays
MAsmtk1_kNN_Abr = np.zeros((nreps, nexp))
MAsmtk1_kNN_Zhao = np.zeros((nreps, nexp))
MAsmtk1_kstarNN = np.zeros((nreps, nexp))
MAsmtk1_GKDE_Sil = np.zeros((nreps, nexp))
MAsmtk1_awkde = np.zeros((nreps, nexp))
MAsmtk1_GKDE_Scott = np.zeros((nreps, nexp))
MAsmtk1_PAk = np.zeros((nreps, nexp))
MAsmtk1_BMTI = np.zeros((nreps, nexp))
MAsmtk1_GMM = np.zeros((nreps, nexp))

smtk1_kNN_Abr = np.zeros((nreps, nexp, 1000))
smtk1_kNN_Zhao = np.zeros((nreps, nexp, 1000))
smtk1_kstarNN = np.zeros((nreps, nexp, 1000))
smtk1_GKDE_Sil = np.zeros((nreps, nexp, 1000))
smtk1_awkde = np.zeros((nreps, nexp, 1000))
smtk1_GKDE_Scott = np.zeros((nreps, nexp, 1000))
smtk1_PAk = np.zeros((nreps, nexp, 1000))
smtk1_BMTI = np.zeros((nreps, nexp, 1000))
smtk1_GMM = np.zeros((nreps, nexp, 1000))


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

        results = compute_roughness(X_k, F_anal_k, d=d, noalign=noalign, maxk = min(X_k.shape[0] - 1, 500))

        # assign results to arrays
        Nsample[i] = results['Nsample']

        MAsmtk1_kNN_Abr[r,i] = results['MAsmtk1_kNN_Abr']
        MAsmtk1_kNN_Zhao[r,i] = results['MAsmtk1_kNN_Zhao']
        MAsmtk1_kstarNN[r,i] = results['MAsmtk1_kstarNN']
        MAsmtk1_GKDE_Sil[r,i] = results['MAsmtk1_GKDE_Sil']
        MAsmtk1_awkde[r,i] = results['MAsmtk1_awkde']
        MAsmtk1_GKDE_Scott[r,i] = results['MAsmtk1_GKDE_Scott']
        MAsmtk1_PAk[r,i] = results['MAsmtk1_PAk']
        MAsmtk1_BMTI[r,i] = results['MAsmtk1_BMTI']
        MAsmtk1_GMM[r,i] = results['MAsmtk1_GMM']

        smtk1_kNN_Abr[r,i] = results['smtk1_kNN_Abr']
        smtk1_kNN_Zhao[r,i] = results['smtk1_kNN_Zhao']
        smtk1_kstarNN[r,i] = results['smtk1_kstarNN']
        smtk1_GKDE_Sil[r,i] = results['smtk1_GKDE_Sil']
        smtk1_awkde[r,i] = results['smtk1_awkde']
        smtk1_GKDE_Scott[r,i] = results['smtk1_GKDE_Scott']
        smtk1_PAk[r,i] = results['smtk1_PAk']
        smtk1_BMTI[r,i] = results['smtk1_BMTI']
        smtk1_GMM[r,i] = results['smtk1_GMM']


        np.savez("results/{}.npz".format(savestring),
            Nsample=Nsample,

            MAsmtk1_kNN_Abr=MAsmtk1_kNN_Abr,
            MAsmtk1_kNN_Zhao=MAsmtk1_kNN_Zhao,
            MAsmtk1_kstarNN=MAsmtk1_kstarNN,
            MAsmtk1_GKDE_Sil=MAsmtk1_GKDE_Sil,
            MAsmtk1_awkde=MAsmtk1_awkde,
            MAsmtk1_GKDE_Scott=MAsmtk1_GKDE_Scott,
            MAsmtk1_PAk=MAsmtk1_PAk,
            MAsmtk1_BMTI=MAsmtk1_BMTI,
            MAsmtk1_GMM=MAsmtk1_GMM,
            
            smtk1_kNN_Abr=smtk1_kNN_Abr,
            smtk1_kNN_Zhao=smtk1_kNN_Zhao,
            smtk1_kstarNN=smtk1_kstarNN,
            smtk1_GKDE_Sil=smtk1_GKDE_Sil,
            smtk1_awkde=smtk1_awkde,
            smtk1_GKDE_Scott=smtk1_GKDE_Scott,
            smtk1_PAk=smtk1_PAk,
            smtk1_BMTI=smtk1_BMTI,
            smtk1_GMM=smtk1_GMM
        )





