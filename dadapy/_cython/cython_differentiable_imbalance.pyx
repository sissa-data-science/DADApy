import numpy as np

cimport numpy as np
from cython.parallel cimport parallel, prange

import cython

from libc.math cimport (  # absolute values for floats, needed when using PBC
    exp,
    fabs,
    nearbyint,
    sqrt,
)


# TODO: @wildromi clean up
# TODO: @wildromi why does this not accept var=var assignments anymore
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def compute_dist_PBC_cython_parallel(double[:, :] X, double[:] box_size, int njobs, bint squared=False):
    """Compute pairwise euclidean distances between points in a periodic boundary condition (PBC) system using Cython with parallel processing.

    Args:
        X : (numpy.ndarray): shape (N, D) of type 'float' (python) a.k.a. 'double' (C).
            The input array of points in a PBC system, where N is the number of points and D is the number of dimensions.
        box_size (numpy.ndarray): shape (D,) of type 'float' (python) a.k.a. 'double' (C)
            The periodicity (size of the PBC box) in each dimension.
        njobs (int): The number of threads to use for parallel processing.
        squared (bool, optional): Whether to return squared distances or regular euclidean distances. Default is False.

    Returns:
        distmatrix: numpy.ndarray, shape (N, N)
            The pairwise distance matrix between points, considering PBC of type 'float' (python) a.k.a. 'double' (C).

    Notes:
        - This function assumes that the input arrays have already been validated for type.
        - The function uses Cython with parallel processing for performance optimization.
    """
    cdef int N = X.shape[0]
    cdef int D = X.shape[1]
    cdef int i, j, k
    cdef double diff, b
    cdef double dist
    distmatrix = np.zeros((N, N), dtype=float, order='C')
    cdef double[:,::1] distmatrix_view = distmatrix

    # Apply periodic boundary conditions to the features of X
    # Set box_size_copy True where period is not zero, False where it is zero.
    # cdef long[:] box_size_copy 
    # box_size_copy = np.ones((D, ), dtype=int)
    # box_size_copy[:] = 1
    # for i in range(D):
    #     if box_size[i] == 0:
    #         box_size_copy[i] = 0
    box_size_copy = [b!=0. for b in box_size]
    X = np.mod(X, box_size, out=np.asarray(X), where=box_size_copy)

    if squared:
        with nogil, parallel(num_threads=njobs):
            for i in prange(N, schedule='static'):
                for j in range(N):
                    dist = 0.0
                    for k in range(D):
                        diff = X[i, k] - X[j, k]
                        if box_size[k] != 0.:
                            # diff=diff/box_size[k]
                            # diff=diff-nearbyint(diff)
                            # diff=diff*box_size[k]
                            if diff > 0.5*box_size[k]:
                                diff = diff - box_size[k]
                            elif diff < -0.5*box_size[k]:
                                diff = diff + box_size[k]
                        dist = dist + diff*diff
                    # Compute pairwise distances
                    distmatrix_view[i, j] = dist
    else:
        with nogil, parallel(num_threads=njobs):
            for i in prange(N, schedule='static'):
                for j in range(N):
                    dist = 0.0
                    for k in range(D):
                        diff = X[i, k] - X[j, k]
                        if box_size[k] != 0.:
                            # diff=diff/box_size[k]
                            # diff=diff-nearbyint(diff)
                            # diff=diff*box_size[k]
                            if diff > 0.5*box_size[k]:
                                diff = diff - box_size[k]
                            elif diff < -0.5*box_size[k]:
                                diff = diff + box_size[k]
                        dist = dist + diff*diff
                    # Compute pairwise distances
                    distmatrix_view[i, j] = sqrt(dist)

    return distmatrix


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def compute_dist_cython_parallel(double[:, :] X, int njobs, bint squared=False):
    """Compute pairwise euclidean distances between points using Cython with parallel processing.

    Args:
        X : (numpy.ndarray): shape (N, D) of type 'float' (python) a.k.a. 'double' (C).
            The input array of points, where N is the number of points and D is the number of dimensions.
        njobs (int): The number of threads to use for parallel processing.
        squared (bool, optional): Whether to return squared distances or regular euclidean distances. Default is False.

    Returns:
        distmatrix: numpy.ndarray, shape (N, N)
            The pairwise distance matrix between points.

    Notes:
        - This function assumes that the input arrays have already been validated for type.
        - The function uses Cython with parallel processing for performance optimization.
    """

    cdef int N = X.shape[0]
    cdef int D = X.shape[1]
    cdef int i, j, k
    cdef double diff
    cdef double dist
    distmatrix = np.zeros((N, N), dtype=float, order='C')
    cdef double[:,::1] distmatrix_view = distmatrix

    if squared:
        with nogil, parallel(num_threads=njobs):
            for i in prange(N, schedule='static'):
                for j in range(N):
                    dist = 0.0
                    for k in range(D):
                        diff = X[i, k] - X[j, k]
                        dist = dist + diff*diff
                    # Compute pairwise distances
                    distmatrix_view[i, j] = dist
    else:
        with nogil, parallel(num_threads=njobs):
            for i in prange(N, schedule='static'):
                for j in range(N):
                    dist = 0.0
                    for k in range(D):
                        diff = X[i, k] - X[j, k]
                        dist = dist + diff*diff
                    # Compute pairwise distances
                    distmatrix_view[i, j] = sqrt(dist)

    return distmatrix


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def return_kernel_imbalance_gradient_cython(double[:,:] dists_rescaled_A not None, double[:,:] data_A not None, long[:,:] rank_matrix_B not None, double[:] gammas not None, double lambd, double[:] period not None, int njobs, bint periodic = False):
    """Compute the gradient of kernel imbalance between input data matrix A and groundtruth data matrix B; Cython implementation.

    Args:
        dists_rescaled_A : numpy.ndarray, shape (N, N), of type 'float' (python) a.k.a. 'double' (C).
            The rescaled distances between points in input array A, where N is the number of points.
        data_A : numpy.ndarray, shape (N, D), of type 'float' (python) a.k.a. 'double' (C).
            The input array A, where N is the number of points and D is the number of dimensions.
        rank_matrix_B : numpy.ndarray, shape (N, N), of type 'int' (python) a.k.a. 'long' (C).
            The rank matrix for groundtruth data array B, where N is the number of points.
        gammas : numpy.ndarray, shape (D,), of type 'float' (python) a.k.a. 'double' (C).
            The array of weight values for the input values, where D is the number of gammas.
            This cannot be initialized to 0's. It can be initialized to all 1 or the inverse of the standard deviation
        lambd : float
            The lambda scaling parameter of the softmax. This can be calculated automatically with python function 'return_optimal_lambda'.
        period : numpy.ndarray
            D(input) periods (input formatted to be periodic starting at 0). If some of the input feature do not have a a period, set those to 0.
            In this cython implementation this must be given, if there is no period read any dummy array of size D.
        njobs : int
            The number of threads to use for parallel processing. 
        periodic : bool
            Whether to use Cython implementation for computing distances. Default is True.

    Returns:
        gradient: numpy.ndarray, shape (D,). The gradient of the kernel imbalance for each variable (dimension).
    """
# This syntax of typing 

    cdef int N, D, i, j, k
    N = data_A.shape[0] 
    D = data_A.shape[1]
    cdef double summ
      
#     # Create memview for the resulting gradient to access it in C-speed (and of other matrices)
    gradient = np.zeros(D, dtype=float)
    cdef double[:] gradient_view = gradient
    
    min_d = np.empty(N, dtype=float)
    cdef double[:] min_dists = min_d
    
    c_m = np.empty((N,N), dtype=float)
    cdef double[:,::1] c_matrix = c_m
    
    alphacol = np.empty((N,1), dtype=float)
    cdef double[:,:] alphacolumn = alphacol

#    if njobs is 0:
#        njobs = multiprocessing.cpu_count()

    if lambd == 0:
        #gradient = np.nan * gradient #why was this set to nan? Does 0 also work? was it to break the program when devide by 0 occured by 0 lambda?
        gradient_view[:] = 0. #!!!!!!!!!!!!!!!!! check that this is indeed ok
    else:
#         # take distance of first nearest neighbor for each point 
#         #min_dists = np.nanmin(dists_rescaled_A, axis=1)[:,np.newaxis]
        #for i in range(N):
        with nogil, parallel(num_threads=njobs):
            for i in prange(N, schedule='static'):
                min_dists[i] = cmin(dists_rescaled_A[i])

    #         # compute the exponential of the negative distances / lambda
    #         # subtraction of minimum distance to avoid overflow problems
    #         # (notice that diagonal elements are already np.nan); 
                summ = 0
                for j in range(N):
                    c_matrix[i,j] = exp(-(dists_rescaled_A[i,j] - min_dists[i])/ lambd)
                    summ = summ + c_matrix[i,j]
    #         # compute c_ij matrix
    #      #   c_matrix = exp_matrix / np.nansum(exp_matrix, axis=1)[:,np.newaxis]
                for j in range(N):
                    c_matrix[i,j] = c_matrix[i,j] / summ
                c_matrix[i,i] = 0 # set diagonale to 0
            
        # compute the gradient term for each gamma (parallelization is faster than the loop below):
        if periodic == True:
            for i in range(D):
                if gammas[i] == 0:
                    gradient_view[i] = 0.
                else:
                    alphacolumn = data_A[:,i,None] # data_A[:,i] creates a 1D vector, the ",None" adds a dimension
                    gradient_view[i] = alphagamma_gradientterm_cython_PBC_parallel(alpha_gamma=i, alphacolumn=alphacolumn, gammas=gammas, period=period, dists_rescaled_A=dists_rescaled_A, rank_matrix_B=rank_matrix_B, c_matrix=c_matrix, njobs=njobs)   
                    gradient_view[i] = (gradient_view[i] * gammas[i]) / (lambd * N*N)
        else:
            for i in range(D):
                if gammas[i] == 0:
                    gradient_view[i] = 0.
                else:
                    alphacolumn = data_A[:,i,None] # data_A[:,i] creates a 1D vector, the ",None" adds a dimension
                    gradient_view[i] = alphagamma_gradientterm_cython_parallel(alpha_gamma=i, alphacolumn=alphacolumn, gammas=gammas, dists_rescaled_A=dists_rescaled_A, rank_matrix_B=rank_matrix_B, c_matrix=c_matrix, njobs=njobs)   
                    gradient_view[i] = (gradient_view[i] * gammas[i]) / (lambd * N*N)    

    return gradient


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef double cmin(double[:] arr) nogil:
    cdef double min = arr[0]
    #cdef double max = -np.inf
    cdef int i
    for i in range(arr.shape[0]):
        if arr[i] < min:
            min = arr[i]
    #    if arr[i] > max
    #        max = arr[i]
    return min #, max


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cpdef double alphagamma_gradientterm_cython_PBC_parallel(int alpha_gamma, double[:,:] alphacolumn, double[:] gammas, double[:] period, double[:,:] dists_rescaled_A, long[:,:] rank_matrix_B, double[:,:] c_matrix, int njobs):
    
    cdef int D,N,i,j
    D = gammas.shape[0]
    N = alphacolumn.shape[0]
    cdef double summ, gradient_alphagamma
    gradient_alphagamma = 0.

    t1 = np.empty((N,N), dtype=float)
    cdef double[:,::1] first_term = t1
    t2 = np.empty(N, dtype=float)
    cdef double[:] second_term = t2
    

    #periodcorrection according to the rescaling factors of the inputs
    cdef double[:] periodalpha
    periodalpha=period[alpha_gamma,None] #this creates a 1D array with just my 1 period number I need
    dists_squared_A = compute_dist_PBC_cython_parallel(alphacolumn, box_size=periodalpha, njobs=njobs, squared=True)
    cdef double [:,:] d_s_A_view = dists_squared_A
    
    #for i in range(N):
    with nogil, parallel(num_threads=njobs):
        for i in prange(N, schedule='static'):
            summ = 0.
            for j in range(N):
                first_term[i,j] = - d_s_A_view[i,j]/dists_rescaled_A[i,j]
                summ = summ + (d_s_A_view[i,j] / dists_rescaled_A[i,j]) * c_matrix[i,j]
            second_term[i] = summ

            #### This following I cannot do because it throughs Nan - probably tue to the parallelization ####
            # for j in range(N): 
            #     gradient_alphagamma = gradient_alphagamma + c_matrix[i,j] * rank_matrix_B[i,j] * (first_term[i,j] + second_term[i])  

            #### Instead I do the weird loop below where I read the intermediate result in the 1D array "second_term" and 
            #### add that up outside the parallelized loop. Looks ugly but is fast ####
            summ=0.
            for j in range(N):
                summ = summ + c_matrix[i,j] * rank_matrix_B[i,j] * (first_term[i,j] + second_term[i])
            second_term[i] = summ   
    for i in range(N):
        gradient_alphagamma = gradient_alphagamma + second_term[i]

    return gradient_alphagamma


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cpdef double alphagamma_gradientterm_cython_parallel(int alpha_gamma, double[:,:] alphacolumn, double[:] gammas, double[:,:] dists_rescaled_A, long[:,:] rank_matrix_B, double[:,:] c_matrix, int njobs):
    
    cdef int D,N,i,j
    D = gammas.shape[0]
    N = alphacolumn.shape[0]
    cdef double summ, gradient_alphagamma    
    gradient_alphagamma = 0.

    t1 = np.empty((N,N), dtype=float)
    cdef double[:,::1] first_term = t1
    t2 = np.empty(N, dtype=float)
    cdef double[:] second_term = t2
    
    cdef double [:,:] d_s_A_view = compute_dist_cython_parallel(alphacolumn, njobs=njobs, squared=True)

    #for i in range(N):
    with nogil, parallel(num_threads=njobs):
        for i in prange(N, schedule='static'):
            summ = 0.
            for j in range(N):
                first_term[i,j] = - d_s_A_view[i,j]/dists_rescaled_A[i,j]
                summ = summ + (d_s_A_view[i,j] / dists_rescaled_A[i,j]) * c_matrix[i,j]
            second_term[i] = summ

            #### This following I cannot do because it throughs Nan - probably tue to the parallelization ####
            # for j in range(N): 
            #     gradient_alphagamma = gradient_alphagamma + c_matrix[i,j] * rank_matrix_B[i,j] * (first_term[i,j] + second_term[i])  

            #### Instead I do the weird loop below where I read the intermediate result in the 1D array "second_term" and 
            #### add that up outside the parallelized loop. Looks ugly but is fast ####
            summ=0.
            for j in range(N):
                summ = summ + c_matrix[i,j] * rank_matrix_B[i,j] * (first_term[i,j] + second_term[i])
            second_term[i] = summ   
    for i in range(N):
        gradient_alphagamma = gradient_alphagamma + second_term[i]
   
    return gradient_alphagamma


