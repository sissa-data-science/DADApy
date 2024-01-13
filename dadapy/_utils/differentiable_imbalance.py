
# Copyright 2021-2023 The DADApy Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Not user-friendly functions for calculating eg. full distance matrix etc."""

from joblib import Parallel, delayed # TODO: might not be necessary
import time
import numpy as np
from sklearn.pairwise import euclidean_distances
from scipy.stats import rankdata

from dadapy._cython import differentiable_imbalance as c_dii

def _compute_optimal_lambda_from_distances(distance_matrix, fraction):
    # sets lambda to the average between the smallest and mean (2nd NN - 1st NN)-distance
    # np.fill_diagonal(distance_matrix, np.nan) ###CHANGE: This I don't need because on the diagonal I have just big values
    NNs = np.sort(distance_matrix, axis=1) #
    min_distances_nn = NNs[:,1] - NNs[:,0]
    return fraction * ((np.min(min_distances_nn) + np.nanmean(min_distances_nn)) / 2)

def _compute_full_dist_matrix(data, period, njobs, cythond=True):
    """Computes the distance matrix based on input data points and optionally a period array.

    Args:
        data (np.ndarray): N x D array containing N datapoints in D-dimensional space.
        period (np.ndarray, optional): D array of periodicity (PBC) for the data points. 
            If None (default), Euclidean distances are computed.
            If a float or numpy.ndarray is provided, PBC distances are computed.
        cythond (bool, optional): Flag indicating whether to use Cython-based distance computation methods. 
            If True (default), Cython-based methods are used if period is not None (otherwise sklearn Euclidean distance).
            If False, Python-based methods are used.
        njobs (int, optional): Number of parallel jobs to use for Cython-based distance computation. 
            Default is 8.

    Returns:
        numpy.ndarray: N x N distance matrix, where N is the number of datapoints in the input data.

    Notes:
        - The function supports both Cython-based and Python-based distance computation methods.
        - If cythond is True and period is not None, the function uses the Cython-based method 'compute_dist_PBC_cython_parallel' with specified box size and parallelization.
        - If cythond is False and period is not None, the function uses the Python-based method 'compute_dist_PBC' with specified box size, maximum k value, and p-value.
        - The diagonal elements of the distance matrix are filled with a value greater than the maximum distance encountered in the matrix, 
            which is necessary for several other functions, including ranking
    """
    # TODO: add the faster python implementation of this (probably sklearn)
    N = data.shape[0]
    D = data.shape[1]

    # Make matrix of distances
    if period is None:
        dist_matrix = euclidean_distances(data)
    else:
        if cythond == True:
            dist_matrix = c_dii.compute_dist_PBC_cython_parallel(data, box_size=period, njobs=njobs)
        else:
            dist_matrix = c_dii.compute_dist_PBC(data, maxk=data.shape[0], box_size=period, p=2)   
    np.fill_diagonal(dist_matrix, np.max(dist_matrix)+1) ###CHANGE - this cannot be 0 because of divide by 0 and not NaN because of cython

    return dist_matrix

def _compute_full_rank_matrix(data, period, njobs, distances=False, cythond=True):
    """Computes the rank matrix based on input data points and optionally a period array, using the 'compute_dist_matrix' function.

    Args:
        data (np.ndarray): N x D array containing N datapoints in D-dimensional space.
        distances (bool, optional): if True, return also the distance matrix. Default: False. 
        period (np.ndarray, optional): D array of periodicity (PBC) for the data points. 
            If None (default), Euclidean distances are computed.
            If a float or numpy.ndarray is provided, PBC distances are computed.
        cythond (bool, optional): Flag indicating whether to use Cython-based distance computation methods. 
            If True (default), Cython-based methods are used if period is not None (otherwise sklearn Euclidean distance).
            If False, Python-based methods are used.

    Returns:
        numpy.ndarray: N x N rank matrix, where N is the number of datapoints in the input data.
    """ 
    # TODO: build better method here, or at least have this only build from ranks
    if period is not None:	
        D = data.shape[1]
        if isinstance(period, np.ndarray) and period.shape == (D,
        ):
            period = period
        elif isinstance(period, (int, float)):
            period = np.full((D), fill_value=period, dtype=float)
        else:
            raise ValueError(
                f"'period' must be either a float scalar or a numpy array of floats of shape ({D},)"
            )

    dist_matrix = _compute_full_dist_matrix(data=data, period=period, cythond=cythond, njobs=njobs)
   # np.fill_diagonal(dist_matrix, np.nan)  ### To give last rank to diag. The distance function already sets diagonal to max, so unnecessary
    # Make rank matrix, notice that ranks go from 1 to N and np.nan elements are ranked N
    rank_matrix = rankdata(dist_matrix, method='average', axis=1).astype(int, copy=False)
    # rank_matrix[rank_matrix == rank_matrix.shape[0]] = np.nan ###CHANGE 
    # # we don't need to set this to nan for it not to count in the sum of kernel imbalance and gradient. Instead, 
    # # we set now the diagonal of the c matrix to 0
    if distances:
        return rank_matrix, dist_matrix
    else:
        return rank_matrix

def _compute_kernel_imbalance(dist_matrix_A, rank_matrix_B, lambd):
    """Computes the kernel imbalance between two matrices based on distances of input data and rank information of groundtruth data.

    Args:
        dist_matrix_A (np.ndarray): N x N array - The distance matrix for between all input data points of input space A. Can
            be computed with 'compute_dist_matrix'
        rank_matrix_B (np.ndarray): N x N rank matrix for the groundtruth data B. 
        lambd (float, optional): The regularization parameter. Default: 0.1. The higher this value, the more nearest neighbors are included. 
            Can be calculated automatically with 'compute_optimal_lambda'. This sets lambda to a distance smaller than the average distance 
            in the data set but bigger than the minimal distance

    Returns:
        kernel_imbalance (float): The computed kernel imbalance value. Please mind that this depends (unlike in the classic information imbalance) 
            on the chosen lambda. To compare several scaled distances compare them using the same lambda, even if they were optimized with different ones.

    Raises:
        None.
    """
    # TODO: clean up
    N = dist_matrix_A.shape[0]
    
    # take distance of first nearest neighbor for each point 
    min_dists = np.min(dist_matrix_A, axis=1)[:,np.newaxis] ###CHANGE do I need nanmin coming from optimization or gradient?
    
    # Make the exponential of the negative distances from the input space / lambda;
    # subtraction of minimum distance does not change c_ij coefficients but avoids
    # overflow problems
    exp_matrix = np.exp(-(dist_matrix_A - min_dists) / lambd)
    
    # Set diagonal elements = nan (i != j), in case dist_matrix_A
    # not obtained with function 'compute_rank_matrix'
    np.fill_diagonal(exp_matrix, 0) ###CHANGE # before I used to set it to nan and do nansum
    
    # compute c_ij matrix
    rowsums = np.sum(exp_matrix, axis=1)[:,np.newaxis]
    c_matrix = exp_matrix / rowsums
    
    # compute kernel imbalance
    kernel_imbalance = 2/N**2 * np.sum(rank_matrix_B * c_matrix)
    
    return kernel_imbalance

def _compute_kernel_imbalance_gradient(dists_rescaled_A, data_A, rank_matrix_B, gammas, lambd, period, njobs, cythond=True):
    """Compute the gradient of kernel imbalance between input data matrix A and groundtruth data matrix B.

    Args:
        dists_rescaled_A : numpy.ndarray, shape (N, N)
            The rescaled distances between points in input array A, where N is the number of points.
        data_A : numpy.ndarray, shape (N, D)
            The input array A, where N is the number of points and D is the number of dimensions.
        rank_matrix_B : numpy.ndarray, shape (N, N)
            The rank matrix for groundtruth data array B, where N is the number of points.
        gammas : numpy.ndarray, shape (D,)
            The array of weight values for the input values, where D is the number of gammas.
            This cannot be initialized to 0's. It can be initialized to all 1 or the inverse of the standard deviation
        lambd : float, optional
            The lambda scaling parameter of the softmax. If None, it is calculated automatically. Default is None.
        period : float or numpy.ndarray/list, optional
            D(input) periods (input formatted to be periodic starting at 0). If not a list, the same period is assumed for all D features
            Default is None, which means no periodic boundary conditions are applied. If some of the input feature do not have a a period, set those to 0.
        njobs : int, optional
            The number of threads to use for parallel processing. Default is None, which uses the maximum number of available CPUs.
        cythond : bool, optional
            Whether to use Cython implementation for computing distances. Default is True.

    Returns:
        gradient: numpy.ndarray, shape (D,). The gradient of the kernel imbalance for each variable (dimension).
    """
    # TODO: Add faster function for python side of this, or remove python entirely.
    # TODO: move typechecks to parent
    N = data_A.shape[0]
    D = data_A.shape[1]
    gradient = np.zeros(D)
    
    if lambd == 0: # TODO: remove type check, this should be handled in class object
        gradient = np.nan * gradient
    else:
        if period is not None:
            if isinstance(period, np.ndarray) and period.shape == (D,
            ):
                period = period
            elif isinstance(period, (int, float)):
                period = np.full((D), fill_value=period, dtype=float)
            else:
                raise ValueError(
                    f"'period' must be either a float scalar or a numpy array of floats of shape ({D},)"
                )

        # take distance of first nearest neighbor for each point 
        min_dists = np.min(dists_rescaled_A, axis=1)[:,np.newaxis] ###CHANGE: do I need nanmin?

        # compute the exponential of the negative distances / lambda 
        # subtraction of minimum distance to avoid overflow problems
        exp_matrix = np.exp(-(dists_rescaled_A - min_dists) / lambd)
        np.fill_diagonal(exp_matrix, 0) ###CHANGE # before I didn't have this line because the diagonal of dists_rescaled_A was nan already

        # compute c_ij matrix
        c_matrix = exp_matrix / np.sum(exp_matrix, axis=1)[:,np.newaxis] ###CHANGE: before nansum

        def alphagamma_gradientterm(alpha_gamma):
            if gammas[alpha_gamma] == 0:
                gradient_alphagamma = 0
            else:
                if period is None:
                    dists_squared_A = euclidean_distances(data_A[:,alpha_gamma,np.newaxis], squared=True)
                else:
                    #periodcorrection according to the rescaling factors of the inputs
                    # start=time.time()
                    periodalpha=period[alpha_gamma]
                    if cythond:
                        dists_squared_A = c_dii.compute_dist_PBC_cython_parallel(data_A[:,alpha_gamma,np.newaxis], box_size=periodalpha[:,np.newaxis], njobs=njobs, squared=True)
                    else:
                        dists_squared_A = np.square(_compute_dist_PBC(data_A[:,alpha_gamma,np.newaxis], maxk=data_A.shape[0], box_size=periodalpha, p=2))
                first_term = - dists_squared_A / dists_rescaled_A
                second_term = np.sum(dists_squared_A / dists_rescaled_A * c_matrix, axis=1)[:,np.newaxis] ###CHAGNE, before nansum
                product_matrix = c_matrix * rank_matrix_B * (first_term + second_term)
                gradient_alphagamma = np.sum(product_matrix)  ###CHANGE, before nansum
            return gradient_alphagamma

        # compute the gradient term for each gamma (parallelization is faster than the loop below):
        gradient_parallel = np.array(Parallel(njobs=njobs, prefer="threads")(delayed(alphagamma_gradientterm)(alpha_gamma) for alpha_gamma in range(len(gammas))))

    gradient = (gradient_parallel * gammas) / (lambd * N**2)

    return gradient

def _optimize_kernel_imbalance(
        groundtruth_data, data, njobs, gammas_0=None, 
        lambd=None, n_epochs=100, l_rate=None, constrain=False, 
        l1_penalty=0., decaying_lr=True, period=None, groundtruthperiod=None, cythond=True
    ):
    """Optimize the differentiable information imbalance using gradient descent of the kernel imbalance between input data matrix A and groundtruth data matrix B.

    Args:
        groundtruth_data : numpy.ndarray, shape (N, Dg)
            The data set used as groundtruth, where N is the number of points and Dg the dimension of it.
        data : numpy.ndarray, shape (N, D)
            The input array A, where N is the number of points and D is the number of dimensions.
        gammas_0 : numpy.ndarray, shape (D,)
            The array of starting weight values for the input values, where D is the dimension of data. If none, it is initialized to 1/var for each variable
            This cannot be initialized to 0's. It can be initialized to all 1 or the inverse of the standard deviation
        lambd : float, optional
            The lambda scaling parameter of the softmax. If None, it is calculated automatically. Default is None.
        n_epochs: int, optional
            The number of epochs in the gradient descent optimization. If None, it is set to 100.
        l_rate: float, optional
            The learning rate of the gradient descent. If None, automatically estimated to be fast.
        constrain: bool
            Constrain the sum of the weights to sum up to the number of weights. Default: False
        l1_penalty: float
            The l1-regularization strength, if sparcity is needed. Default: 0 (l1-regularization turned off).
        decaying_lr: bool
            Use exponentially decaying learning rate in gradient descent or not. Default: True.
        period : float or numpy.ndarray/list, optional  
            D(input data) periods (input should be formatted to be periodic starting at 0). If not a list, the same period is assumed for all D features
            Default is None, which means no periodic boundary conditions are applied. If some of the input feature do not have a a period, set those to 0.
        groudntruthperiod : float or numpy.ndarray/list, optional  
            Dg(groundtruth_data) periods (groundtruth_data should be formatted to be periodic starting at 0). If not a list, the same period is assumed for all Dg features
            Default is None, which means no periodic boundary conditions are applied. If some of the input feature do not have a a period, set those to 0.
        njobs : int, optional
            The number of threads to use for parallel processing. Default is None, which uses the maximum number of available CPUs.
        cythond : bool, optional
            Whether to use Cython implementation for computing distances. Default is True.

    Returns:
        gammas_list, kernel_imbalances,l1_penalties
        gammas_list: np.ndarray, shape (n_epochs, D). List of lists of all weights for each feature for each step in the optimization
        kernel_imbalances: np.ndarray, shape (n_epochs, ). List of the differentiable information imbalances during the optimization
        l1_penalties: np.ndarray, shape (n_epochs, ). List of the l1_penaltie terms that were added to the imbalances in the loss function

    """
    # TODO: This function is quite complicated with handling stuff like the period and the learning rate
    # Discuss with @wildromi and move typechecks and such to class object
    #  gammacheck = 0
    N = data.shape[0]
    D = data.shape[1]
    
    kernel_imbalances = np.ones(n_epochs+1) # +1: to include initial value
    l1_penalties = np.zeros(n_epochs+1)
    gammas_list=np.zeros((n_epochs+1, D))
    scaling=1 #if there is no constraint on rescaling of gammas
    
    rank_matrix_B = _compute_full_rank_matrix(groundtruth_data, period=groundtruthperiod, cythond=cythond)

    # initializations
    if constrain:
        scaling = 1/np.max(np.abs(gammas_0))

    gammas = scaling*gammas_0
    gammas_list[0]=gammas

    # rescale input data with the weights
    rescaled_data_A = gammas * data
    # for adaptive lambda: calculate distance matrix in rescaled input
    if period is not None:
        # Removed period typecheck here, this should be handled somewhere else
        dists_rescaled_A = _compute_full_dist_matrix(data=rescaled_data_A, period=gammas*period, cythond=cythond, njobs=njobs)
    else:
        periodarray=None
        dists_rescaled_A = _compute_full_dist_matrix(data=rescaled_data_A, period=None, cythond=cythond, njobs=njobs)

    if lambd is not None:
        lambd = scaling*lambd # to make the gradient scale invariant. adaptive lambda automatically scales lambda to the features for scale invariance
        adaptive_lambd = False
    else:
        adaptive_lambd = True
        lambd = _compute_optimal_lambda_from_distances(dists_rescaled_A)
    
    kernel_imbalances[0] = _compute_kernel_imbalance(dists_rescaled_A, rank_matrix_B, lambd)
    l1_penalties[0] = l1_penalty * np.sum(np.abs(gammas))
    lrate= l_rate # for not expon. decaying learning rates

    for i_epoch in range(n_epochs):
        # compute gradient * SCALING!!!! to be scale invariant
        if not cythond:
            gradient = _compute_kernel_imbalance_gradient(dists_rescaled_A, data, rank_matrix_B, gammas, lambd, period=period) * scaling
        else:
            if period is not None:
                periodic = True
                myperiod = period
            else:
                periodic = False
                myperiod = gammas*0. # dummy array, not used in cython:
            gradient = c_dii.compute_kernel_imbalance_gradient_cython(dists_rescaled_A, data, rank_matrix_B, gammas, lambd, period=myperiod, njobs=njobs, periodic=periodic) * scaling 
        if np.isnan(gradient).any(): # If any of the gradient elements turned to nan
            kernel_imbalances[i_epoch+1] = kernel_imbalances[i_epoch]
            l1_penalties[i_epoch+1] = l1_penalties[i_epoch]
            gammas_list[i_epoch+1] = gammas_list[i_epoch]
            print("At least one gradient element turned to Nan, no optimization possible.")
            break
        else:
            # exponentially decaying lr
            if decaying_lr == True:
                lrate = l_rate * 2**(-i_epoch/10)  # every 10 epochs the learning rate will be halfed
            
            # Gradient Descent Clipping update (Tsuruoka 2008)
            # update rescaling weights, making sure they do not do a sign change due to learning rate step    
            gammas_new = gammas - lrate * gradient
            for i,gam in enumerate(gammas_new):
                if gam > 0:
                    gammas_new[i] = max(0., gam -lrate*l1_penalty)
                elif gam < 0:
                    gammas_new[i] = np.abs(min(0., gam +lrate*l1_penalty))
            gammas = gammas_new
            # exit the loop if all weights are 0 (e.g. l1-regularization too strong)
            if gammas.any() == 0:
                kernel_imbalances[i_epoch+1] = kernel_imbalances[i_epoch]
                l1_penalties[i_epoch+1] = l1_penalties[i_epoch]
                gammas_list[i_epoch+1] = gammas_list[i_epoch]
                print("The l1-regularization of ",l1_penalty," is too high. All features would be set to 0. No full optimization possible")    
                break

            # apply constrain on the weights
            if constrain:
                scaling = 1/np.max(np.abs(gammas))
            gammas = scaling*gammas    

            # for adaptive lambda: calculate distance matrix in rescaled input
            rescaled_data_A = gammas * data
            if period is not None:
                periodarray=gammas*period
            dists_rescaled_A = _compute_full_dist_matrix(rescaled_data_A, period=periodarray, cythond=cythond, njobs=njobs)
            lambd = scaling*lambd # to make the gradient scale invariant. adaptive lambda automatically scales lambda to the features for scale invariance
            if adaptive_lambd:
                lambd = _compute_optimal_lambda_from_distances(dists_rescaled_A)

            # compute kernel imbalance
            kernel_imbalances[i_epoch+1] = _compute_kernel_imbalance(dists_rescaled_A, rank_matrix_B, lambd)
            l1_penalties[i_epoch+1] = l1_penalty * np.sum(np.abs(gammas))
            gammas_list[i_epoch+1] = gammas
    #  if gammacheck == 1:
    #      print("The l1-regularization of ",l1_penalty," is too high. All features set to 0. No optimization possible")    
    if l1_penalty == 0.:
        return gammas_list, kernel_imbalances, kernel_imbalances*0
    else:
        return gammas_list, kernel_imbalances, l1_penalties 

def _compute_optimal_learning_rate(
        groundtruth_data, data, gammas_0, lambd, 
        n_epochs=50, constrain=False, l1_penalty=0.0, decaying_lr=True, 
        period=None, groundtruthperiod=None, nsamples=300, lr_list=None
    ):
    
    """Do a stepwise backward eliminitaion of features and after each elimination GD otpmize the kernel imblance
    Args:
        groundtruth_data (np.ndarray): N x D(groundtruth) array containing N datapoints in all the groundtruth features D(groundtruth)
        data (np.ndarray): N x D(input) array containing N datapoints in D input features whose weights are optimized to reproduce the groundtruth distances 
        gammas_0 (np.ndarray or list): D(input) initial weights for the input features. No zeros allowed here
        lambd (float): softmax scaling. If None (preferred) this chosen automatically with compute_optimial_lambda 
        n_epochs (int): number of epochs in each optimization cycle
        constrain (bool): if True, rescale the weights so the biggest weight = 1 
        l1_penalty (float): l1 regularization strength
        decaying_lr (bool): default: True. Apply decaying learning rate = l_rate * 2**(-i_epoch/10) - every 10 epochs the learning rate will be halfed
        period (float or np.ndarray/list): D(input) periods (input formatted to be 0-period). If not a list, the same period is assumed for all D features
        groundtruthperiod (float or np.ndarray/list): D(groundtruth) periods (groundtruth formatted to be 0-period). If not a list, the same period is assumed for all D(groundtruth) features
        nsamples (int): Number of samples to use for the learning rate screening. Default = 300.
               lr_list (np.ndarray or list): learning rates to try
    Returns:
        opt_l_rate (float): Learning rate, which leads to optimal unregularized (no l1-penalty) result in the specified number of epochs
        kernel_imbalances_list: values of the kernel imbalance during optimization in n_epochs using the l_rate. Plot to ensure the optimization went well
    """
    # TODO: smaller standard value for nsamples here?
    stride = int(np.round(len(data)/nsamples))
    # TODO: why not randomly select here?
    in_data = data[::stride]
    groundtruth = groundtruth_data[::stride]
    if lr_list==None: 
        # TODO: np.logspace here?
        lrates=[0.001, 0.01,  0.1, 1. ,10., 50., 100., 200.]
    else:
        lrates=lr_list
    gs = np.zeros((len(lrates),n_epochs+1,in_data.shape[1]))
    ks = np.zeros((len(lrates),n_epochs+1))
    
    # optmizations for different learning rates
    for i, lrate in enumerate(lrates):
        gs[i], ks[i], _ = _optimize_kernel_imbalance(groundtruth_data=groundtruth, data=in_data, gammas_0=gammas_0, lambd=lambd,
            n_epochs=n_epochs, l_rate=lrate, constrain=constrain, l1_penalty=l1_penalty, decaying_lr=decaying_lr, 
            period=period, groundtruthperiod=groundtruthperiod, njobs=None, cythond=True
        )
    
    # find best imbalance
    opt_lrate_index = np.nanargmin(ks[:,-1])
    opt_l_rate = lrates[opt_lrate_index]
    kernel_imbalances_list = ks[opt_lrate_index]
    
    return opt_l_rate, kernel_imbalances_list

def _optimize_kernel_imbalance_static_zeros(groundtruth_data, data, gammas_0, njobs, lambd=None, n_epochs=100, l_rate=0.1, constrain=False, decaying_lr=True, period=None, groundtruthperiod=None, cythond=True):
    """Optimization where 0 weights stay 0. Used in backward eliminitaion of features
    Args:
        groundtruth_data (np.ndarray): N x D(groundtruth) array containing N datapoints in all the groundtruth features D(groundtruth)
        data (np.ndarray): N x D(input) array containing N datapoints in D input features whose weights are optimized to reproduce the groundtruth distances
        gammas_0 (np.ndarray or list): D(input) initial weights for the input features. No zeros allowed here
        lambd (float): softmax scaling. If None (preferred) this chosen automatically with compute_optimial_lambda
        n_epochs (int): number of epochs in each optimization cycle
        l_rate (float): learning rate. Has to be tuned, especially if constrain=True (otherwise optmization could fail)
        constrain (bool): if True, rescale the weights so the biggest weight = 1
        l1_penalty (float): l1 regularization strength
        decaying_lr (bool): default: True. Apply decaying learning rate = l_rate * 2**(-i_epoch/10) - every 10 epochs the learning rate will be halfed
        period (float or np.ndarray/list): D(input) periods (input formatted to be 0-period). If not a list, the same period is assumed for all D features
        groundtruthperiod (float or np.ndarray/list): D(groundtruth) periods (groundtruth formatted to be 0-period). 
                                                      If not a list, the same period is assumed for all D(groundtruth) features
    Returns:
        gammas:
        kernel_imbalances:
    """
    # batch GD optimization with zeroes staying zeros - needed for eliminate_backward_greedy_kernel_imbalance

    N = data.shape[0]
    D = data.shape[1]

    if period is not None:
        if isinstance(period, np.ndarray) and period.shape == (D,
        ):
            period = period
        elif isinstance(period, (int, float)):
            period = np.full((D), fill_value=period, dtype=float)
        else:
            raise ValueError(
                f"'period' must be either a float scalar or a numpy array of floats of shape ({D},)"
            )

    kernel_imbalances = np.ones(n_epochs+1) # +1: to include initial value
    gammas_list=np.zeros((n_epochs+1,D))
    scaling=1 #if there is no constraint on rescaling of gammas
    rank_matrix_B = _compute_full_rank_matrix(groundtruth_data, period=groundtruthperiod, cythond=cythond)

    # initializations
    if constrain:
        scaling = 1/np.max(np.abs(gammas_0))
    gammas = scaling*gammas_0
    gammas_list[0]=gammas

    # rescale input data with the weights
    rescaled_data_A = gammas * data

    # for adaptive lambda: calculate distance matrix in rescaled input
    # for adaptive lambda: calculate distance matrix in rescaled input
    dists_rescaled_A = _compute_full_dist_matrix(data=rescaled_data_A, period=gammas*period, cythond=cythond, njobs=njobs)

    if lambd is not None:
        lambd = scaling*lambd # to make the gradient scale invariant. adaptive lambda automatically scales lambda to the features for scale invariance
        adaptive_lambd = False
    elif lambd is None:
        adaptive_lambd = True
        lambd = _compute_optimal_lambda_from_distances(dists_rescaled_A)

    kernel_imbalances[0] = _compute_kernel_imbalance(dists_rescaled_A, rank_matrix_B, lambd)
    lrate= l_rate # for not expon. decaying learning rates

    for i_epoch in range(n_epochs):

        # compute gradient * SCALING!!!! to be scale invariant
        # compute gradient * SCALING!!!! to be scale invariant
        if cythond == False:
            gradient = _compute_kernel_imbalance_gradient(dists_rescaled_A, data, rank_matrix_B, gammas, lambd, period=period, cythond=False) * scaling
        else:
            if period is not None:
                periodic = True
                myperiod = period
            else:
                periodic = False
                myperiod = gammas*0
            gradient = c_dii.compute_kernel_imbalance_gradient_cython(dists_rescaled_A, data, rank_matrix_B, gammas, lambd, period=period, njobs=njobs, periodic=periodic) * scaling 
        if np.isnan(gradient).any(): # If any of the gradient elements turned to nan
            kernel_imbalances[i_epoch+1] = kernel_imbalances[i_epoch]
            gammas_list[i_epoch+1] = gammas_list[i_epoch]
            print("At least one gradient element turned to Nan, no optimization possible.")
            break
        else:
            #set gradient to 0 if gamma was 0, so the new gamma is also 0. DIFFERENT FROM REGULAR OPTIMIZATION
            gradient[gammas == 0] = 0

            # exponentially decaying lr
            if decaying_lr == True:
                lrate = l_rate * 2**(-i_epoch/10)  # every 10 epochs the learning rate will be halfed

            # Gradient Descent Clipping update (Tsuruoka 2008) - only works with l1 penalty... otherwise we do not reach 0
            # update rescaling weights, making sure they do not do a sign change due to learning rate step
            gammas_new = gammas - lrate * gradient
            for i,gam in enumerate(gammas_new):
                if gam > 0:
                    gammas_new[i] = max(0., gam)
                elif gam < 0:
                    gammas_new[i] = np.abs(min(0., gam))
            ## don't use this line: it makes the performance also without lasso worse:
            #gammas[gammas_new < gammas] = gammas_new[gammas_new < gammas] #only accept steps that actually make the current weight smaller
            ## use instead:
            gammas = gammas_new

            # apply constrain on the weights
            if constrain:
                scaling = 1/np.max(np.abs(gammas))
            gammas = scaling*gammas

            # for adaptive lambda: calculate distance matrix in rescaled input
            rescaled_data_A = gammas * data
            if period is not None:
                periodarray=gammas*period
            dists_rescaled_A = _compute_full_dist_matrix(rescaled_data_A, period=periodarray,cythond=cythond, njobs=njobs)
            lambd = scaling*lambd # to make the gradient scale invariant. adaptive lambda automatically scales lambda to the features for scale invariance
            if adaptive_lambd:
                lambd = _compute_optimal_lambda_from_distances(dists_rescaled_A)

            # compute kernel imbalance
            kernel_imbalances[i_epoch+1] = _compute_kernel_imbalance(dists_rescaled_A, rank_matrix_B, lambd)
            gammas_list[i_epoch+1] = gammas

    return gammas_list, kernel_imbalances

def _refine_lasso_optimization(
        gs, ks, ls, l1_penalties, groundtruth_data, data, gammas_0, njobs,
        lambd=None, n_epochs=50, l_rate=None, constrain=False, 
        decaying_lr=True, period=None, groundtruthperiod=None, cythond=True
    ):
    """Genereate more lasso runs in between lasso strengths that produced non-consecutive numbers of non-zero weights
    Args:
        gs (np.ndarray): n_lassos x n_epochs x D array containing the weights during the optimization for each lasso strength in the previous optimization
        ks (np.ndarray): n_lassos x n_epochs array containing the information imbalance term of the loss in the previous optimization
        ls (np.ndarray): n_lassos x n_epochs array containing the l1-term of the loss in the previous optimization
        l1_penalties (list): list of n_lassos l1-regularization strengths tested in the previous optimization
        groundtruth_data (np.ndarray): N x D(groundtruth) array containing N datapoints in all the groundtruth features D(groundtruth)
        data (np.ndarray): N x D(input) array containing N datapoints in D input features whose weights are optimized to reproduce the groundtruth distances
        gammas_0 (np.ndarray or list): D(input) initial weights for the input features. No zeros allowed here
        lambd (float): softmax scaling. If None (preferred) this chosen automatically with compute_optimial_lambda
        l_rate (float or None): if None, the learning rate is determined automatically with optimize_learning_rate
        n_epochs (int): number of epochs in each optimization cycle
        constrain (bool): if True, rescale the weights so the biggest weight = 1
        decaying_lr (bool): default: True. Apply decaying learning rate = l_rate * 2**(-i_epoch/10) - every 10 epochs the learning rate will be halfed
        period (float or np.ndarray/list): D(input) periods (input formatted to be 0-period). If not a list, the same period is assumed for all D features
        groundtruthperiod (float or np.ndarray/list): D(groundtruth) periods (groundtruth formatted to be 0-period). If not a list, the same period is assumed for all D(groundtruth) features
    Returns:
        opt_l_rate (float): Learning rate, which leads to optimal unregularized (no l1-penalty) result in the specified number of epochs
        kernel_imbalances_list: values of the kernel imbalance during optimization in n_epochs using the l_rate. Plot to ensure the optimization went well
    """
    # TODO: cleanup
    # Find where to refine the lasso and decide on new l1 penalties
    gs[np.isnan(gs)] = 0
    l0gs = np.linalg.norm(gs[:,-1,:],0, axis=1)
    start=data.shape[1]
    refinement_needed = []
    if l0gs[0] == start:
        highest=1*start
        for i, l0 in enumerate(l0gs):
            if l0 < highest-1:
                refinement_needed.append([i-1,i,int((l0gs[i-1]-l0)*2)]) #lower and upper limit of refinement and how many points to add
            if l0 < highest:
                highest = l0
    else: print("starting lasso too big")

    newpenalties=[]
    for i in range(len(refinement_needed)):
        newpenalties.append(list(np.linspace(l1_penalties[refinement_needed[i][0]], l1_penalties[refinement_needed[i][1]], refinement_needed[i][2])[1:-1]))


    # Optimimize the kernel imbalance for lassos that were missing, intercollate with the old data

    all_l1s= l1_penalties[0:refinement_needed[0][0]+1] #add old l1's until the first refinement
    all_gs = list(gs[0:refinement_needed[0][0]+1])
    all_ks = list(ks[0:refinement_needed[0][0]+1])
    all_ls = list(ls[0:refinement_needed[0][0]+1])

    for i, newl1 in enumerate(newpenalties):

        gs_new = np.zeros((len(newl1),n_epochs+1,data.shape[1]))
        ks_new = np.zeros((len(newl1),n_epochs+1))
        ls_new = np.zeros((len(newl1),n_epochs+1))
        # do the new optimizations
        for j in range(len(newl1)):
            start = time.time()
            gs_new[j], ks_new[j], ls_new[j] = _optimize_kernel_imbalance(groundtruth_data=groundtruth_data, data=data, gammas_0=gammas_0, lambd=lambd,
                                            n_epochs=n_epochs, l_rate=l_rate, constrain=constrain, l1_penalty=newl1[j],decaying_lr=decaying_lr, 
                                            period=period, groundtruthperiod=groundtruthperiod, njobs=njobs, cythond=cythond)
            end = time.time()
            print("in intercollation ",i+1," of ",len(newpenalties), "for test l1 ",j+1," of ",len(newl1),", the time was: ",end - start)

        # make the intercollated list of penalties
        all_l1s = all_l1s + newpenalties[i]  #add refinement 
        all_gs =  all_gs  + list(gs_new)
        all_ks =  all_ks  + list(ks_new)
        all_ls =  all_ls  + list(ls_new)
        if i < len(newpenalties)-1:
            all_l1s = all_l1s + l1_penalties[refinement_needed[i][1]:refinement_needed[i+1][0]+1] #add old l1s after refinement until next refinement
            all_gs =  all_gs  + list(gs[refinement_needed[i][1]:refinement_needed[i+1][0]+1])
            all_ks =  all_ks  + list(ks[refinement_needed[i][1]:refinement_needed[i+1][0]+1])
            all_ls =  all_ls  + list(ls[refinement_needed[i][1]:refinement_needed[i+1][0]+1])
        else:
            all_l1s = all_l1s + l1_penalties[refinement_needed[i][1]:] #add old l1s after last refinement
            all_gs =  all_gs  + list(gs[refinement_needed[i][1]:])
            all_ks =  all_ks  + list(ks[refinement_needed[i][1]:])
            all_ls =  all_ls  + list(ls[refinement_needed[i][1]:])
    return np.array(all_gs), np.array(all_ks), np.array(all_ls), all_l1s