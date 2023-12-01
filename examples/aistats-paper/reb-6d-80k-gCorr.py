import matplotlib.pyplot as plt
plt.rc('figure', max_open_warning = 0) #no limit for fugures
plt.rcParams.update({                  #use lateX fonts
  "text.usetex": True,
  "font.family": "Helvetica"
})

from dadapy import *
from dadapy._utils.utils import _align_arrays
from scipy import stats
from scipy.ndimage.filters import gaussian_filter1d
import time
import numpy as np

from awkde import GaussianKDE
#%matplotlib notebook

def gaussian(x, mu, sig):
   return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))/np.sqrt(2*np.pi)


#KILLER GRAPH
## Define pdf analytically

def den(v):
    #harmonic potential = 6*(x_i)**2
    
    value = 1.
    
    for i in range(2,6):
        value *= np.exp(-6*v[i]*v[i]);
    
    r = value*pow( 2.*np.exp(-(-1.5 + v[0])*(-1.5 + v[0]) - (-2.5 + v[1])*(-2.5 + v[1])) + 3*np.exp(-2*v[0]*v[0] - 0.25*v[1]*v[1]) , 3 )
    #r = value*pow( 2.*np.exp(-(-1.5 + v[1])*(-1.5 + v[1]) - (-2.5 + v[0])*(-2.5 + v[0])) + 3*np.exp(-2*v[1]*v[1] - 0.25*v[0]*v[0]) , 3 )

    return r


def free(v):
    
    return - np.log(den(v))



#X80k = np.genfromtxt('datasets/6d_double_well-100k.txt')[40000:, 1:]


# get current folder path
import os

# import 6d_double_well-100k.txt in the current path
X80k = np.genfromtxt('datasets/6d_double_well-100k.txt')[40000:, 1:]


print(len(X80k))


F_anal80k = []
for i, x in enumerate(X80k):
    F_anal80k.append(free(x))
F_anal80k = np.array(F_anal80k)

nexp = 10


Nsample = np.zeros(nexp,dtype=np.int32)
ksel_Abr = np.zeros(nexp,dtype=np.int32)
ksel_Zhao = np.zeros(nexp,dtype=np.int32)
h_Sil = np.zeros(nexp)

MAE_kNN_Abr = np.zeros(nexp)
MAE_kNN_Zhao = np.zeros(nexp)
MAE_kstarNN = np.zeros(nexp)
MAE_PAk = np.zeros(nexp)
MAE_BMTI = np.zeros(nexp)
MAE_GKDE_Sil = np.zeros(nexp)

MSE_kNN_Abr = np.zeros(nexp)
MSE_kNN_Zhao = np.zeros(nexp)
MSE_kstarNN = np.zeros(nexp)
MSE_PAk = np.zeros(nexp)
MSE_BMTI = np.zeros(nexp)
MSE_GKDE_Sil = np.zeros(nexp)

time_kNN_Abr = np.zeros(nexp)
time_kNN_Zhao = np.zeros(nexp)
time_kstarNN = np.zeros(nexp)
time_PAk = np.zeros(nexp)
time_BMTI = np.zeros(nexp)
time_GKDE_Sil = np.zeros(nexp)


## Per il production

for i in reversed(range(0,nexp)):
    Xk=X80k[::2**i]
    Nsample[i] = len(Xk)
    print(Nsample[i])


for i in reversed(range(0,nexp)):
    
    # create dataset
    Xk=X80k[::2**i]
    Nsample[i] = len(Xk)
    
    # F_anal
    F_anal_k = []
    for j, x in enumerate(Xk):
        F_anal_k.append(free(x))
    F_anal_k = np.array(F_anal_k)
    
    # init dataset
    data = Data(Xk,verbose=False)
    data.compute_distances(maxk = min(Xk.shape[0] - 1, 100))
    d=6
    data.set_id(d)
    print()
    print("Nsample:")
    print(data.N)
    
    #kNN_Abr
    sec = time.perf_counter()
    ksel_Abr[i]=Nsample[i]**(4./(4.+d))
    data.compute_density_kNN(ksel_Abr[i])
    time_kNN_Abr[i] = time.perf_counter() - sec
    off_k, F_k = _align_arrays(-data.log_den,data.log_den_err,F_anal_k)
    MAE_kNN_Abr[i] = np.mean(np.abs(F_k-F_anal_k))
    MSE_kNN_Abr[i] = np.mean((F_k-F_anal_k)**2)
    
    #kNN_Zhao
    sec = time.perf_counter()
    ksel_Zhao[i]=Nsample[i]**(2./(2.+d))
    data.compute_density_kNN(ksel_Zhao[i])
    time_kNN_Zhao[i] = time.perf_counter() - sec
    off_k, F_k = _align_arrays(-data.log_den,data.log_den_err,F_anal_k)
    MAE_kNN_Zhao[i] = np.mean(np.abs(F_k-F_anal_k))
    MSE_kNN_Zhao[i] = np.mean((F_k-F_anal_k)**2)
    
    #kstarNN
    sec = time.perf_counter()
    data.compute_density_kstarNN()
    time_kstarNN[i] = time.perf_counter() - sec
    off_k, F_k = _align_arrays(-data.log_den,data.log_den_err,F_anal_k)
    MAE_kstarNN[i] = np.mean(np.abs(F_k-F_anal_k))
    MSE_kstarNN[i] = np.mean((F_k-F_anal_k)**2)

    #PAk
    sec = time.perf_counter()
    data.compute_density_PAk()
    time_PAk[i] = time.perf_counter() - sec
    off_k, F_k = _align_arrays(-data.log_den,data.log_den_err,F_anal_k)
    MAE_PAk[i] = np.mean(np.abs(F_k-F_anal_k))
    MSE_PAk[i] = np.mean((F_k-F_anal_k)**2)

    #BMTI
    sec = time.perf_counter()
    data.compute_density_gCorr_OLD()
    time_BMTI[i] = time.perf_counter() - sec
    off_k, F_k = _align_arrays(-data.log_den,data.log_den_err,F_anal_k)
    MAE_BMTI[i] = np.mean(np.abs(F_k-F_anal_k))
    MSE_BMTI[i] = np.mean((F_k-F_anal_k)**2)

    #GKDE Silverman
    sec = time.perf_counter()
    kdesil = GaussianKDE(glob_bw="silverman", alpha=0.0, diag_cov=True)
    kdesil.fit(data.X)
    h_Sil[i]=kdesil.glob_bw
    F_gksil = - np.log(kdesil.predict(data.X))
    time_GKDE_Sil[i] = time.perf_counter() - sec
    off_gksil, F_gksil = _align_arrays(F_gksil,np.ones_like(F_anal_k),F_anal_k)
    MAE_GKDE_Sil[i] = np.mean(np.abs(F_gksil-F_anal_k))
    MSE_GKDE_Sil[i] = np.mean((F_gksil-F_anal_k)**2)
    


    print("kNN_Abr",ksel_Abr[i],": ",MAE_kNN_Abr[i])
    print("kNN_Zhao",ksel_Zhao[i],": ",MAE_kNN_Zhao[i])
    print("kstarNN: ",MAE_kstarNN[i])
    print("PAk: ",MAE_PAk[i])
    print("BMTI: ",MAE_BMTI[i])
    print("GKDE Silverman: ",MAE_GKDE_Sil[i])
    

    np.savetxt("reb-MAE-6d-80k.txt",np.column_stack((
        Nsample,
        MAE_kNN_Abr,
        MAE_kNN_Zhao,
        MAE_kstarNN,
        MAE_PAk,
        MAE_BMTI,
        MAE_GKDE_Sil,
        ksel_Abr,
        ksel_Zhao,
        h_Sil
    )),fmt='%8f')

    np.savetxt("reb-MSE-6d-80k.txt",np.column_stack((
        Nsample,
        MSE_kNN_Abr,
        MSE_kNN_Zhao,
        MSE_kstarNN,
        MSE_PAk,
        MSE_BMTI,
        MSE_GKDE_Sil
    )),fmt='%8f')

    np.savetxt("reb-times-6d-80k.txt",np.column_stack((
        Nsample,
        time_kNN_Abr,
        time_kNN_Zhao,
        time_kstarNN,
        time_PAk,
        time_BMTI,
        time_GKDE_Sil
    )),fmt='%8f')



# for i in reversed(range(0,nexp)):
    
#     # create dataset
#     Xk=X80k[::2**i]
#     Nsample[i] = len(Xk)
    
#     # F_anal
#     F_anal_k = []
#     for j, x in enumerate(Xk):
#         F_anal_k.append(free(x))
#     F_anal_k = np.array(F_anal_k)
    
#     # init dataset
#     data = Data(Xk,verbose=False)
#     data.compute_distances(maxk = 100)
#     d=6
#     data.set_id(d)
#     print()
#     print("Nsample:")
#     print(data.N)
        
#     #GKDE Scott
#     prova_h = stats.gaussian_kde(data.X.T)
#     h_Scott[i] = prova_h.scotts_factor()
#     F_gkscott = - prova_h.logpdf(data.X.T)
#     off_k, F_k = _align_arrays(- prova_h.logpdf(data.X.T),data.log_den_err,F_anal_k)
#     MAE_GKDE_Scott[i] = np.mean(np.abs(F_k-F_anal_k))


#     print("GKDE: ",MAE_GKDE_Scott[i])
    
    
#     np.savetxt("killer-MAE-6d-80k.txt",np.column_stack((
#         MAE_kNN_Abr,
#         MAE_kNN_Zhao,
#         MAE_kstarNN,
#         MAE_PAk,
#         MAE_BMTI,
#         MAE_GKDE_Scott,
#         Nsample,
#         ksel_Abr,
#         ksel_Zhao,
#         h_Scott
#     )),fmt='%8f')



# F_anal80k = []
# for i, x in enumerate(X80k):
#     F_anal80k.append(free(x))
# F_anal80k = np.array(F_anal80k)

# nexp = 10

# MAE_GKDE_Sil = np.zeros(nexp)
# Nsample = np.zeros(nexp,dtype=np.int32)
# h_Sil = np.zeros(nexp)

# ## Per il production

# for i in reversed(range(0,nexp)):
#     Xk=X80k[::2**i]
#     Nsample[i] = len(Xk)
#     print(Nsample[i])


# for i in reversed(range(0,nexp)):
    
#     # create dataset
#     Xk=X80k[::2**i]
#     Nsample[i] = len(Xk)
    
#     # F_anal
#     F_anal_k = []
#     for j, x in enumerate(Xk):
#         F_anal_k.append(free(x))
#     F_anal_k = np.array(F_anal_k)
    
#     # init dataset
#     data = Data(Xk,verbose=False)
#     data.compute_distances(maxk = 100)
#     d=6
#     data.set_id(d)
#     print()
#     print("Nsample:")
#     print(data.N)
    
#     #GKDE Silverman
#     kdesil = GaussianKDE(glob_bw="silverman", alpha=0.0, diag_cov=True)
#     kdesil.fit(data.X)
#     h_Sil[i]=kdesil.glob_bw
#     F_gksil = - np.log(kdesil.predict(data.X))
#     off_gksil, F_gksil = _align_arrays(F_gksil,np.ones_like(F_anal_k),F_anal_k)
#     MAE_GKDE_Sil[i] = np.mean(np.abs(F_gksil-F_anal_k))

#     print("GKDE Silverman: ",MAE_GKDE_Sil[i])

    
#     np.savetxt("killer-MAE-6d-80k-GKDE_Silverman.txt",np.column_stack((
#         Nsample,
#         MAE_GKDE_Sil,
#         h_Sil
#     )),fmt='%8f')