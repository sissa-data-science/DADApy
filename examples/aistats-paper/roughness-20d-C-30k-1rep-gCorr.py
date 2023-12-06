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

savestring="roughness-20d-C-30k-simple_align-1rep"
noalign=False

# import dataset
#X_full = np.genfromtxt('datasets/6d_double_well-1.2M-last_400k.txt')[::8] #keep 50k
X_full = np.genfromtxt('datasets/20D_and_gt_dataset_panel_C_Fig3.txt')[:, :20]


#print(X_full.shape) 

d = 7

F_full = np.genfromtxt('datasets/20D_and_gt_dataset_panel_C_Fig3.txt')[:, 20]
#F_full = np.array([free_6d(x) for x in X_full])


print("Dataset size: ",X_full.shape[0])

nreps = 1 # number of repetitions
print("Number of repetitions: ",nreps)
nexp = 1 # number of dataset sizes

# create nreps random subsets of the 
nsample = 30000
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
MArelrk1_kNN_Abr = np.zeros((nreps, nexp))
MArelrk1_kNN_Zhao = np.zeros((nreps, nexp))
MArelrk1_kstarNN = np.zeros((nreps, nexp))
MArelrk1_GKDE_Sil = np.zeros((nreps, nexp))
MArelrk1_awkde = np.zeros((nreps, nexp))
MArelrk1_GKDE_Scott = np.zeros((nreps, nexp))
MArelrk1_PAk = np.zeros((nreps, nexp))
MArelrk1_BMTI = np.zeros((nreps, nexp))
MArelrk1_GMM = np.zeros((nreps, nexp))

KLDrk1_kNN_Abr = np.zeros((nreps, nexp))
KLDrk1_kNN_Zhao = np.zeros((nreps, nexp))
KLDrk1_kstarNN = np.zeros((nreps, nexp))
KLDrk1_GKDE_Sil = np.zeros((nreps, nexp))
KLDrk1_awkde = np.zeros((nreps, nexp))
KLDrk1_GKDE_Scott = np.zeros((nreps, nexp))
KLDrk1_PAk = np.zeros((nreps, nexp))
KLDrk1_BMTI = np.zeros((nreps, nexp))
KLDrk1_GMM = np.zeros((nreps, nexp))

rk1_anal = np.zeros((nreps, nexp, 10000))
rk1_kNN_Abr = np.zeros((nreps, nexp, 10000))
rk1_kNN_Zhao = np.zeros((nreps, nexp, 10000))
rk1_kstarNN = np.zeros((nreps, nexp, 10000))
rk1_GKDE_Sil = np.zeros((nreps, nexp, 10000))
rk1_awkde = np.zeros((nreps, nexp, 10000))
rk1_GKDE_Scott = np.zeros((nreps, nexp, 10000))
rk1_PAk = np.zeros((nreps, nexp, 10000))
rk1_BMTI = np.zeros((nreps, nexp, 10000))
rk1_GMM = np.zeros((nreps, nexp, 10000))

relrk1_kNN_Abr = np.zeros((nreps, nexp, 10000))
relrk1_kNN_Zhao = np.zeros((nreps, nexp, 10000))
relrk1_kstarNN = np.zeros((nreps, nexp, 10000))
relrk1_GKDE_Sil = np.zeros((nreps, nexp, 10000))
relrk1_awkde = np.zeros((nreps, nexp, 10000))
relrk1_GKDE_Scott = np.zeros((nreps, nexp, 10000))
relrk1_PAk = np.zeros((nreps, nexp, 10000))
relrk1_BMTI = np.zeros((nreps, nexp, 10000))
relrk1_GMM = np.zeros((nreps, nexp, 10000))

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

        MArelrk1_kNN_Abr[r,i] = results['MArelrk1_kNN_Abr']
        MArelrk1_kNN_Zhao[r,i] = results['MArelrk1_kNN_Zhao']
        MArelrk1_kstarNN[r,i] = results['MArelrk1_kstarNN']
        MArelrk1_GKDE_Sil[r,i] = results['MArelrk1_GKDE_Sil']
        MArelrk1_awkde[r,i] = results['MArelrk1_awkde']
        MArelrk1_GKDE_Scott[r,i] = results['MArelrk1_GKDE_Scott']
        MArelrk1_PAk[r,i] = results['MArelrk1_PAk']
        MArelrk1_BMTI[r,i] = results['MArelrk1_BMTI']
        MArelrk1_GMM[r,i] = results['MArelrk1_GMM']

        KLDrk1_kNN_Abr[r,i] = results['KLDrk1_kNN_Abr']
        KLDrk1_kNN_Zhao[r,i] = results['KLDrk1_kNN_Zhao']
        KLDrk1_kstarNN[r,i] = results['KLDrk1_kstarNN']
        KLDrk1_GKDE_Sil[r,i] = results['KLDrk1_GKDE_Sil']
        KLDrk1_awkde[r,i] = results['KLDrk1_awkde']
        KLDrk1_GKDE_Scott[r,i] = results['KLDrk1_GKDE_Scott']
        KLDrk1_PAk[r,i] = results['KLDrk1_PAk']
        KLDrk1_BMTI[r,i] = results['KLDrk1_BMTI']
        KLDrk1_GMM[r,i] = results['KLDrk1_GMM']

        rk1_anal[r,i] = results['rk1_anal']
        rk1_kNN_Abr[r,i] = results['rk1_kNN_Abr']
        rk1_kNN_Zhao[r,i] = results['rk1_kNN_Zhao']
        rk1_kstarNN[r,i] = results['rk1_kstarNN']
        rk1_GKDE_Sil[r,i] = results['rk1_GKDE_Sil']
        rk1_awkde[r,i] = results['rk1_awkde']
        rk1_GKDE_Scott[r,i] = results['rk1_GKDE_Scott']
        rk1_PAk[r,i] = results['rk1_PAk']
        rk1_BMTI[r,i] = results['rk1_BMTI']
        rk1_GMM[r,i] = results['rk1_GMM']

        relrk1_kNN_Abr[r,i] = results['relrk1_kNN_Abr']
        relrk1_kNN_Zhao[r,i] = results['relrk1_kNN_Zhao']
        relrk1_kstarNN[r,i] = results['relrk1_kstarNN']
        relrk1_GKDE_Sil[r,i] = results['relrk1_GKDE_Sil']
        relrk1_awkde[r,i] = results['relrk1_awkde']
        relrk1_GKDE_Scott[r,i] = results['relrk1_GKDE_Scott']
        relrk1_PAk[r,i] = results['relrk1_PAk']
        relrk1_BMTI[r,i] = results['relrk1_BMTI']
        relrk1_GMM[r,i] = results['relrk1_GMM']

        np.savez("results/{}.npz".format(savestring),
            Nsample=Nsample,

            MArelrk1_kNN_Abr=MArelrk1_kNN_Abr,
            MArelrk1_kNN_Zhao=MArelrk1_kNN_Zhao,
            MArelrk1_kstarNN=MArelrk1_kstarNN,
            MArelrk1_GKDE_Sil=MArelrk1_GKDE_Sil,
            MArelrk1_awkde=MArelrk1_awkde,
            MArelrk1_GKDE_Scott=MArelrk1_GKDE_Scott,
            MArelrk1_PAk=MArelrk1_PAk,
            MArelrk1_BMTI=MArelrk1_BMTI,
            MArelrk1_GMM=MArelrk1_GMM,

            KLDrk1_kNN_Abr=KLDrk1_kNN_Abr,
            KLDrk1_kNN_Zhao=KLDrk1_kNN_Zhao,
            KLDrk1_kstarNN=KLDrk1_kstarNN,
            KLDrk1_GKDE_Sil=KLDrk1_GKDE_Sil,
            KLDrk1_awkde=KLDrk1_awkde,
            KLDrk1_GKDE_Scott=KLDrk1_GKDE_Scott,
            KLDrk1_PAk=KLDrk1_PAk,
            KLDrk1_BMTI=KLDrk1_BMTI,
            KLDrk1_GMM=KLDrk1_GMM,

            rk1_anal=rk1_anal,
            rk1_kNN_Abr=rk1_kNN_Abr,
            rk1_kNN_Zhao=rk1_kNN_Zhao,
            rk1_kstarNN=rk1_kstarNN,
            rk1_GKDE_Sil=rk1_GKDE_Sil,
            rk1_awkde=rk1_awkde,
            rk1_GKDE_Scott=rk1_GKDE_Scott,
            rk1_PAk=rk1_PAk,
            rk1_BMTI=rk1_BMTI,
            rk1_GMM=rk1_GMM,
            
            relrk1_kNN_Abr=relrk1_kNN_Abr,
            relrk1_kNN_Zhao=relrk1_kNN_Zhao,
            relrk1_kstarNN=relrk1_kstarNN,
            relrk1_GKDE_Sil=relrk1_GKDE_Sil,
            relrk1_awkde=relrk1_awkde,
            relrk1_GKDE_Scott=relrk1_GKDE_Scott,
            relrk1_PAk=relrk1_PAk,
            relrk1_BMTI=relrk1_BMTI,
            relrk1_GMM=relrk1_GMM
        )