import cython
import numpy as np

cimport numpy as np

from scipy.special import gammaln

from libc.math cimport exp, log, pi, pow

DTYPE = np.int
floatTYPE = np.float
boolTYPE = np.bool

ctypedef np.int_t DTYPE_t
ctypedef np.float64_t floatTYPE_t

# ----------------------------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.cdivision(True)
def _compute_kstar(floatTYPE_t id_sel,
                    DTYPE_t Nele,
                    DTYPE_t maxk,
                    floatTYPE_t Dthr,#=23.92812698,
                    np.ndarray[DTYPE_t, ndim = 2] dist_indices,
                    np.ndarray[floatTYPE_t, ndim = 2] distances
):


    cdef floatTYPE_t dL, vvi, vvj
    cdef DTYPE_t i, j, ksel
    cdef np.ndarray[DTYPE_t, ndim = 1] kstar = np.empty(Nele, dtype=int)
    cdef floatTYPE_t prefactor = exp( id_sel / 2.0 * log(pi) - gammaln((id_sel + 2.0) / 2.0) )

    for i in range(Nele):
        j = 4
        dL = 0.0
        while j < maxk and dL < Dthr:
            ksel = j - 1
            vvi = prefactor * pow(distances[i, ksel], id_sel)
            vvj = prefactor * pow(distances[dist_indices[i, j], ksel], id_sel)
            dL = -2.0 * ksel * ( log(vvi) + log(vvj) - 2.0 * log(vvi + vvj) + log(4) )
            j = j + 1
        if j == maxk:
            kstar[i] = j - 1
        else:
            kstar[i] = j - 2

    return kstar


@cython.boundscheck(False)
@cython.cdivision(True)
def _compute_kstar_interp(floatTYPE_t id_sel,
                          DTYPE_t Nele,
                          DTYPE_t maxk,
                          floatTYPE_t Dthr,  #=23.92812698,
                          np.ndarray[DTYPE_t, ndim = 2] cross_dist_indices,
                          np.ndarray[floatTYPE_t, ndim = 2] cross_distances,
                          np.ndarray[floatTYPE_t, ndim = 2] data_distances
                          ):


    cdef floatTYPE_t dL, vvi, vvj
    cdef DTYPE_t i, j, ksel
    cdef np.ndarray[DTYPE_t, ndim = 1] kstar = np.empty(Nele, dtype=int)
    cdef floatTYPE_t prefactor = exp( id_sel / 2.0 * log(pi) - gammaln((id_sel + 2.0) / 2.0) )

    for i in range(Nele):
        j = 4
        dL = 0.0
        while j < maxk and dL < Dthr:
            ksel = j - 1
            vvi = prefactor * pow(cross_distances[i, ksel], id_sel)
            vvj = prefactor * pow(data_distances[cross_dist_indices[i, j], ksel], id_sel)
            dL = -2.0 * ksel * ( log(vvi) + log(vvj) - 2.0 * log(vvi + vvj) + log(4) )
            j = j + 1
        if j == maxk:
            kstar[i] = j - 1
        else:
            kstar[i] = j - 2

    return kstar



# ----------------------------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.cdivision(True)
def _return_Gaussian_kde_periodic(  np.ndarray[floatTYPE_t, ndim = 2] X_source,
                                    np.ndarray[floatTYPE_t, ndim = 2] Y_sample,
                                    np.ndarray[floatTYPE_t, ndim = 1] smoothing_parameter,
                                    np.ndarray[floatTYPE_t, ndim = 1] period
):

    cdef DTYPE_t i, j, dim
    cdef floatTYPE_t temp, temp2, normalisation, sm2
    
    cdef DTYPE_t nx = X_source.shape[0]
    cdef DTYPE_t ny = Y_sample.shape[0]
    cdef DTYPE_t dims = X_source.shape[1]

    cdef floatTYPE_t sqrt2pi = pow(2*np.pi,0.5)

    cdef np.ndarray[floatTYPE_t, ndim = 1] density = np.zeros((ny,),dtype=np.float_)

    for j in range(nx):
        normalisation = nx*pow(sqrt2pi*smoothing_parameter[j],dims)
        sm2 = smoothing_parameter[j]*smoothing_parameter[j]
        for i in range(ny):
            temp2 = 0.
            for dim in range(dims):
                temp = X_source[j, dim] - Y_sample[i, dim]
                if temp > period[dim]/2.:
                    temp -= period[dim]
                if temp < -period[dim]/2.:
                    temp += period[dim]
                temp2 += temp*temp
            density[i]+=exp( -0.5*temp2/sm2 )/ normalisation

    return density
# ----------------------------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.cdivision(True)
def _return_Gaussian_kde(   np.ndarray[floatTYPE_t, ndim = 2] X_source,
                            np.ndarray[floatTYPE_t, ndim = 2] Y_sample,
                            np.ndarray[floatTYPE_t, ndim = 1] smoothing_parameter
):

    cdef DTYPE_t i, j, dim
    cdef floatTYPE_t temp, temp2, normalisation, sm2
    
    cdef DTYPE_t nx = X_source.shape[0]
    cdef DTYPE_t ny = Y_sample.shape[0]
    cdef DTYPE_t dims = X_source.shape[1]

    cdef floatTYPE_t sqrt2pi = pow(2*np.pi,0.5)

    cdef np.ndarray[floatTYPE_t, ndim = 1] density = np.zeros((ny,),dtype=np.float_)

    for j in range(nx):
        normalisation = nx*pow(sqrt2pi*smoothing_parameter[j],dims)
        sm2 = smoothing_parameter[j]*smoothing_parameter[j]
        for i in range(ny):
            temp2 = 0.
            for dim in range(dims):
                temp = X_source[j, dim] - Y_sample[i, dim]
                temp2 += temp*temp
            density[i]+=exp( -0.5*temp2/sm2 )/ normalisation

    return density

# ----------------------------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.cdivision(True)
def _return_gradient_Gaussian_kde_periodic( np.ndarray[floatTYPE_t, ndim = 2] X_source,
                                            np.ndarray[floatTYPE_t, ndim = 2] Y_sample,
                                            np.ndarray[floatTYPE_t, ndim = 1] smoothing_parameter,
                                            np.ndarray[floatTYPE_t, ndim = 1] period
):

    cdef DTYPE_t i, j, dim
    cdef floatTYPE_t temp, temp2, normalisation, sm2
    
    cdef DTYPE_t nx = X_source.shape[0]
    cdef DTYPE_t ny = Y_sample.shape[0]
    cdef DTYPE_t dims = X_source.shape[1]

    cdef floatTYPE_t sqrt2pi = pow(2*np.pi,0.5)

    cdef np.ndarray[floatTYPE_t, ndim = 1] tmpvec = np.zeros((dims,),dtype=np.float_)
    cdef np.ndarray[floatTYPE_t, ndim = 1] density = np.zeros((ny,),dtype=np.float_)
    cdef np.ndarray[floatTYPE_t, ndim = 2] gradient = np.zeros((ny,dims),dtype=np.float_)

    for j in range(nx):
        normalisation = nx*pow(sqrt2pi*smoothing_parameter[j],dims)
        sm2 = smoothing_parameter[j]*smoothing_parameter[j]
        for i in range(ny):
            temp2 = 0.
            for dim in range(dims):
                temp = X_source[j, dim] - Y_sample[i, dim]
                if temp > period[dim]/2.:
                    temp -= period[dim]
                if temp < -period[dim]/2.:
                    temp += period[dim]
                temp2 += temp*temp
                tmpvec[dim]=temp
            #density[i]+=exp( -0.5*temp2/sm2 )/ normalisation
            # if I do not return the density I can neglect the normalisation
            density[i]+=exp( -0.5*temp2/sm2 )
            for dim in range(dims):
                #gradient[i,dim] -= tmpvec[dim]/sm2*exp( -0.5*temp2/sm2 )/ normalisation
                gradient[i,dim] -= tmpvec[dim]/sm2*exp( -0.5*temp2/sm2 )
    for i in range(ny):
        for dim in range(dims):
            gradient[i,dim] /= density[i]

    return gradient

# ----------------------------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.cdivision(True)
def _return_gradient_Gaussian_kde(  np.ndarray[floatTYPE_t, ndim = 2] X_source,
                                    np.ndarray[floatTYPE_t, ndim = 2] Y_sample,
                                    np.ndarray[floatTYPE_t, ndim = 1] smoothing_parameter
):

    cdef DTYPE_t i, j, dim
    cdef floatTYPE_t temp, temp2, normalisation, sm2
    
    cdef DTYPE_t nx = X_source.shape[0]
    cdef DTYPE_t ny = Y_sample.shape[0]
    cdef DTYPE_t dims = X_source.shape[1]

    cdef floatTYPE_t sqrt2pi = pow(2*np.pi,0.5)

    cdef np.ndarray[floatTYPE_t, ndim = 1] tmpvec = np.zeros((dims,),dtype=np.float_)
    cdef np.ndarray[floatTYPE_t, ndim = 1] density = np.zeros((ny,),dtype=np.float_)
    cdef np.ndarray[floatTYPE_t, ndim = 2] gradient = np.zeros((ny,dims),dtype=np.float_)

    for j in range(nx):
        normalisation = nx*pow(sqrt2pi*smoothing_parameter[j],dims)
        sm2 = smoothing_parameter[j]*smoothing_parameter[j]
        for i in range(ny):
            temp2 = 0.
            for dim in range(dims):
                temp = X_source[j, dim] - Y_sample[i, dim]
                temp2 += temp*temp
                tmpvec[dim]=temp
            #density[i]+=exp( -0.5*temp2/sm2 )/ normalisation
            # if I do not return the density I can neglect the normalisation
            density[i]+=exp( -0.5*temp2/sm2 )
            for dim in range(dims):
                #gradient[i,dim] -= tmpvec[dim]/sm2*exp( -0.5*temp2/sm2 )/ normalisation
                gradient[i,dim] -= tmpvec[dim]/sm2*exp( -0.5*temp2/sm2 )
    for i in range(ny):
        for dim in range(dims):
            gradient[i,dim] /= density[i]

    return gradient
    
# ----------------------------------------------------------------------------------------------