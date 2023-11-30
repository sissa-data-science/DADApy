from dadapy import *
import time
import numpy as np


X = np.genfromtxt('/home/matteo/Dottorato/density_estimation_shared/datasets/6d_double_well.txt')[2000:, 1:]
X = X[::10]
data = Data(X,verbose=True)
data.compute_distances(maxk = 800)
data.set_id(6)
#from _cython._debug_cython_grads.debug_cython_grads import provaprova
from _cython._debug_cython_grads.debug_cython_grads_2 import return_deltaFs_cross_covariance, provaprova
data.compute_pearson(comp_p_mat=True)
data.compute_deltaFs_grads_semisum()
smallnumber = 1.e-10
data.grads_var += smallnumber*np.tile(np.eye(data.dims),(data.N,1,1))

sec = time.time()
#inv_Gamma = return_deltaFs_cross_covariance( data.grads_var,
inv_Gamma = provaprova( data.grads_var,
            data.neigh_vector_diffs,
            data.nind_list,
            data.pearson_mat,
            data.Fij_var_array
)
print("{0:0.2f} seconds to carry out the computation.".format(time.time() - sec))