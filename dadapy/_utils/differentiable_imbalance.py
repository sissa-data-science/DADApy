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

import time
from functools import wraps
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from scipy.spatial import cKDTree  # TODO: remove if removing compute_dist_PBC
from scipy.stats import rankdata
from sklearn.metrics.pairwise import euclidean_distances

from dadapy._cython import cython_differentiable_imbalance as c_dii

CYTHON_DTYPE = np.float64


def cast_ndarrays(func):
    """Convert any float np.ndarray to the datatype that cython accepts.

    Args:
        func (callable): function to be decorated

    Returns:
        func (callable): function with all float type np.ndarrays converted to CYTHON_TYPE
    """

    @wraps(func)
    def cast_wrapped(*args, **kwargs):
        args = (
            (
                arg.astype(CYTHON_DTYPE)
                if (isinstance(arg, np.ndarray) and arg.dtype.kind == "f")
                else arg
            )
            for arg in args
        )
        for key, value in kwargs.items():
            if isinstance(value, np.ndarray) and (value.dtype.kind == "f"):
                kwargs[key] = value.astype(CYTHON_DTYPE)
        return func(*args, **kwargs)

    return cast_wrapped


def _return_dist_PBC(X, maxk, box_size=None, p=2, cutoff=np.inf):
    """Compute the neighbours of each point taking into account periodic boundaries conditions and eventual cutoff
    Args:
        X (np.ndarray): array of dimension N x D
        maxk (int): number of neighbours to save
        box_size (float, np.ndarray(float)): sizes of PBC walls. Single value is interpreted as cubic box.
        p (int): Minkowski p-norm used
        cutoff (float): set an upper bound to the distances. Over such threshold a np.inf will occur
    Returns:
        distmatrix (np.ndarray(float)): N x maxk (maxk=usually N) array containing the distances from each point to the first maxk nn
    """
    # TODO: @wildromi is there a reason we don't use dadapy._utils.compute_NN_PBC here?
    box_size_copy = [
        i != 0.0 for i in box_size
    ]  # get a mask for potential 0-period variables
    X = np.mod(
        X, box_size, out=X, where=box_size_copy
    )  # wrap all variables around the period (modulus), except the 0-period ones
    tree = cKDTree(X, boxsize=box_size)
    dist, ind = tree.query(X, k=maxk, p=p, distance_upper_bound=cutoff)
    # resorts the NN-sorted indices back into order and select the according distances
    distmatrix = np.take_along_axis(dist, np.argsort(ind, axis=1), axis=1)
    return distmatrix


def _return_optimal_lambda_from_distances(distance_matrix, fraction=1.0):
    # sets lambda to the average between the smallest and mean (2nd NN - 1st NN)-distance
    NNs = np.sort(distance_matrix, axis=1)  #
    min_distances_nn = NNs[:, 1] - NNs[:, 0]
    return fraction * ((np.min(min_distances_nn) + np.nanmean(min_distances_nn)) / 2)


@cast_ndarrays
def _return_full_dist_matrix(
    data: np.ndarray, n_jobs: int, period: np.ndarray = None, cythond=True
):
    """Computes the distance matrix based on input data points and optionally a period array.

    Args:
        data (np.ndarray): N x D array containing N datapoints in D-dimensional space.
        period (np.ndarray, optional): D array of periodicity (PBC) for the data points.
            If None (default), Euclidean distances are computed.
            If a float or numpy.ndarray is provided, PBC distances are computed.
        cythond (bool, optional): Flag indicating whether to use Cython-based distance computation methods.
            If True (default), Cython-based methods are used if period is not None (otherwise sklearn Euclidean distance).
            If False, Python-based methods are used.
        n_jobs (int, optional): Number of parallel jobs to use for Cython-based distance computation.
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
    # Make matrix of distances
    if period is None:
        dist_matrix = euclidean_distances(data)
    else:
        if cythond:
            # print(data, n_jobs, period, cythond)
            dist_matrix = c_dii.compute_dist_PBC_cython_parallel(data, period, n_jobs)
        else:
            dist_matrix = _return_dist_PBC(
                data, maxk=data.shape[0], box_size=period, p=2
            )
    np.fill_diagonal(
        dist_matrix, np.max(dist_matrix) + 1
    )  # this cannot be 0 because of divide by 0 and not NaN because of cython

    return dist_matrix


def _return_full_rank_matrix(
    data, n_jobs, period: np.ndarray = None, distances=False, cythond=True
):
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

    dist_matrix = _return_full_dist_matrix(
        data=data, period=period, cythond=cythond, n_jobs=n_jobs
    )
    # np.fill_diagonal(dist_matrix, np.nan)  ### To give last rank to diag. The distance function already sets diagonal to max, so unnecessary
    # Make rank matrix, notice that ranks go from 1 to N and np.nan elements are ranked N
    rank_matrix = rankdata(dist_matrix, method="average", axis=1).astype(
        int, copy=False
    )
    if distances:
        return rank_matrix, dist_matrix
    else:
        return rank_matrix


def _return_dii(dist_matrix_A, rank_matrix_B, lambd):
    """Computes the DII between two matrices based on distances of input data and rank information of groundtruth data.

    Args:
        dist_matrix_A (np.ndarray): N x N array - The distance matrix for between all input data points of input space A. Can
            be computed with 'compute_dist_matrix'
        rank_matrix_B (np.ndarray): N x N rank matrix for the groundtruth data B.
        lambd (float, optional): The regularization parameter. Default: 0.1. The higher this value, the more nearest neighbors are included.
            Can be calculated automatically with 'return_optimal_lambda'. This sets lambda to a distance smaller than the average distance
            in the data set but bigger than the minimal distance

    Returns:
        dii (float): The computed DII value. Please mind that this depends (unlike in the classic information imbalance)
            on the chosen lambda. To compare several scaled distances compare them using the same lambda, even if they were optimized with different ones.

    Raises:
        None.
    """
    N = dist_matrix_A.shape[0]

    # take distance of first nearest neighbor for each point
    min_dists = np.min(dist_matrix_A, axis=1)[:, np.newaxis]

    # Make the exponential of the negative distances from the input space / lambda;
    # subtraction of minimum distance does not change c_ij coefficients but avoids
    # overflow problems
    exp_matrix = np.exp(-(dist_matrix_A - min_dists) / lambd)

    # Set diagonal elements = nan (i != j), in case dist_matrix_A
    # not obtained with function 'compute_rank_matrix'
    np.fill_diagonal(exp_matrix, 0)

    # compute c_ij matrix
    rowsums = np.sum(exp_matrix, axis=1)[:, np.newaxis]
    c_matrix = exp_matrix / rowsums

    # compute DII
    dii = 2 / N**2 * np.sum(rank_matrix_B * c_matrix)

    return dii


def _return_dii_gradient_python(
    dists_rescaled_A: np.ndarray,
    data_A: np.ndarray,
    rank_matrix_B: np.ndarray,
    weights,
    lambd: float,
    n_jobs: int,
    period: np.ndarray = None,
):
    """Compute the gradient of DII between input data matrix A and groundtruth data matrix B.

    Args:
        dists_rescaled_A : numpy.ndarray, shape (N, N)
            The rescaled distances between points in input array A, where N is the number of points.
        data_A : numpy.ndarray, shape (N, D)
            The input array A, where N is the number of points and D is the number of dimensions.
        rank_matrix_B : numpy.ndarray, shape (N, N)
            The rank matrix for groundtruth data array B, where N is the number of points.
        weights : numpy.ndarray, shape (D,)
            The array of weight values for the input values, where D is the number of weights.
            This cannot be initialized to 0's. It can be initialized to all 1 or the inverse of the standard deviation
        lambd : float, optional
            The lambda scaling parameter of the softmax. If None, it is calculated automatically. Default is None.
        period : float or numpy.ndarray/list, optional
            D(input) periods (input formatted to be periodic starting at 0). If not a list, the same period is assumed for all D features
            Default is None, which means no periodic boundary conditions are applied. If some of the input feature do not have a a period: np.ndarray=None set those to 0.
        n_jobs : int, optional
            The number of threads to use for parallel processing. Default is None, which uses the maximum number of available CPUs.

    Returns:
        gradient: numpy.ndarray, shape (D,). The gradient of the DII for each variable (dimension).
    """

    N = data_A.shape[0]
    D = data_A.shape[1]
    gradient = np.zeros(D, dtype=CYTHON_DTYPE)

    if lambd == 0:
        gradient = np.nan * gradient
    else:
        if period is not None:
            if isinstance(period, np.ndarray) and period.shape == (D,):
                period = period
            elif isinstance(period, (int, float)):
                period = np.full((D), fill_value=period, dtype=CYTHON_DTYPE)
            else:
                raise ValueError(
                    f"'period' must be either a float scalar or a numpy array of floats of shape ({D},)"
                )

        # take distance of first nearest neighbor for each point
        min_dists = np.min(dists_rescaled_A, axis=1)[:, np.newaxis]

        # compute the exponential of the negative distances / lambda
        # subtraction of minimum distance to avoid overflow problems
        exp_matrix = np.exp(-(dists_rescaled_A - min_dists) / lambd)
        np.fill_diagonal(exp_matrix, 0)

        # compute c_ij matrix
        c_matrix = exp_matrix / np.sum(exp_matrix, axis=1)[:, np.newaxis]

        def alphaweight_gradientterm(alpha_weight):
            if weights[alpha_weight] == 0:
                gradient_alphaweight = 0
            else:
                if period is None:
                    dists_squared_A = euclidean_distances(
                        data_A[:, alpha_weight, np.newaxis], squared=True
                    )
                else:
                    # periodcorrection according to the rescaling factors of the inputs
                    periodalpha = period[alpha_weight]
                    dists_squared_A = np.square(
                        _return_dist_PBC(
                            data_A[:, alpha_weight, np.newaxis],
                            maxk=data_A.shape[0],
                            box_size=[periodalpha],
                            p=2,
                        )
                    )
                first_term = -dists_squared_A / dists_rescaled_A
                second_term = np.sum(
                    dists_squared_A / dists_rescaled_A * c_matrix, axis=1
                )[:, np.newaxis]
                product_matrix = c_matrix * rank_matrix_B * (first_term + second_term)
                gradient_alphaweight = np.sum(product_matrix)
            return gradient_alphaweight

        # compute the gradient term for each weight (parallelization is faster):
        gradient_parallel = np.array(
            Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(alphaweight_gradientterm)(alpha_weight)
                for alpha_weight in range(len(weights))
            )
        )

    gradient = (gradient_parallel * weights) / (lambd * N**2)

    return gradient


@cast_ndarrays
def _return_dii_gradient(
    dists_rescaled_A: np.ndarray,
    data_A: np.ndarray,
    rank_matrix_B: np.ndarray,
    weights,
    lambd: float,
    n_jobs: int,
    period: np.ndarray = None,
    cythond: bool = True,
):
    """Compute the gradient of DII between input data matrix A and groundtruth data matrix B.

    Args:
        dists_rescaled_A : numpy.ndarray, shape (N, N)
            The rescaled distances between points in input array A, where N is the number of points.
        data_A : numpy.ndarray, shape (N, D)
            The input array A, where N is the number of points and D is the number of dimensions.
        rank_matrix_B : numpy.ndarray, shape (N, N)
            The rank matrix for groundtruth data array B, where N is the number of points.
        weights : numpy.ndarray, shape (D,)
            The array of weight values for the input values, where D is the number of weights.
            This cannot be initialized to 0's. It can be initialized to all 1 or the inverse of the standard deviation
        lambd : float, optional
            The lambda scaling parameter of the softmax. If None, it is calculated automatically. Default is None.
        period : float or numpy.ndarray/list, optional
            D(input) periods (input formatted to be periodic starting at 0). If not a list, the same period is assumed for all D features
            Default is None, which means no periodic boundary conditions are applied. If some of the input feature do not have a a period: np.ndarray=None set those to 0.
        n_jobs : int, optional
            The number of threads to use for parallel processing. Default is None, which uses the maximum number of available CPUs.
        cythond : bool, optional
            Whether to use Cython implementation for computing distances. Default is True.

    Returns:
        gradient: numpy.ndarray, shape (D,). The gradient of the DII for each variable (dimension).
    """
    if not cythond:
        return _return_dii_gradient_python(
            dists_rescaled_A=dists_rescaled_A,
            data_A=data_A,
            rank_matrix_B=rank_matrix_B,
            weights=weights,
            lambd=lambd,
            n_jobs=n_jobs,
            period=period,
        )
    else:
        if period is not None:
            periodic = True
            myperiod = period
        else:
            periodic = False
            myperiod = (
                weights * 0.0
            )  # dummy array, but necessary for cython. 0-periods are not used in the cython function.

        return c_dii.return_dii_gradient_cython(
            dists_rescaled_A,
            data_A,
            rank_matrix_B,
            weights,
            lambd,
            myperiod,
            n_jobs,
            periodic,
        )


@cast_ndarrays
def _optimize_dii(
    groundtruth_data: np.ndarray,
    data: np.ndarray,
    n_jobs: int,
    weights_0: np.ndarray = None,
    lambd: float = None,
    n_epochs: int = 100,
    l_rate: float = None,
    constrain: bool = False,
    l1_penalty: float = 0.0,
    decaying_lr: bool = True,
    period: np.ndarray = None,
    groundtruthperiod: np.ndarray = None,
    cythond: bool = True,
):
    """Optimize the differentiable information imbalance using gradient descent of the DII between input data matrix A and groundtruth data matrix B.

    Args:
        groundtruth_data : numpy.ndarray, shape (N, Dg)
            The data set used as groundtruth, where N is the number of points and Dg the dimension of it.
        data : numpy.ndarray, shape (N, D)
            The input array A, where N is the number of points and D is the number of dimensions.
        weights_0 : numpy.ndarray, shape (D,)
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
        n_jobs : int, optional
            The number of threads to use for parallel processing. Default is None, which uses the maximum number of available CPUs.
        cythond : bool, optional
            Whether to use Cython implementation for computing distances. Default is True.

    Returns:
        weights_list, diis,l1_penalties
        weights_list: np.ndarray, shape (n_epochs, D). List of lists of all weights for each feature for each step in the optimization
        diis: np.ndarray, shape (n_epochs, ). List of the differentiable information imbalances during the optimization
        l1_penalties: np.ndarray, shape (n_epochs, ). List of the l1_penaltie terms that were added to the imbalances in the loss function

    """
    #  weightcheck = 0
    N = data.shape[0]
    D = data.shape[1]

    diis = np.ones(n_epochs + 1)  # +1: to include initial value
    l1_penalties = np.zeros(n_epochs + 1)
    weights_list = np.zeros((n_epochs + 1, D))
    scaling = 1  # if there is no constraint on rescaling of weights

    rank_matrix_B = _return_full_rank_matrix(
        groundtruth_data, n_jobs=n_jobs, period=groundtruthperiod, cythond=cythond
    )
    # initializations
    if constrain:
        scaling = 1 / np.max(np.abs(weights_0))

    weights = scaling * weights_0
    weights_list[0] = weights

    # rescale input data with the weights
    rescaled_data_A = weights * data
    # for adaptive lambda: calculate distance matrix in rescaled input
    if period is not None:
        # Removed period typecheck here, this should be handled somewhere else
        dists_rescaled_A = _return_full_dist_matrix(
            data=rescaled_data_A,
            period=weights * period,
            cythond=cythond,
            n_jobs=n_jobs,
        )
    else:
        periodarray = None
        dists_rescaled_A = _return_full_dist_matrix(
            data=rescaled_data_A, period=None, cythond=cythond, n_jobs=n_jobs
        )

    if lambd is not None:
        lambd = (
            scaling * lambd
        )  # to make the gradient scale invariant. adaptive lambda automatically scales lambda to the features for scale invariance
        adaptive_lambd = False
    else:
        adaptive_lambd = True
        lambd = _return_optimal_lambda_from_distances(dists_rescaled_A)

    diis[0] = _return_dii(dists_rescaled_A, rank_matrix_B, lambd)
    l1_penalties[0] = l1_penalty * np.sum(np.abs(weights))
    lrate = l_rate  # for not expon. decaying learning rates

    for i_epoch in range(n_epochs):
        # compute gradient * SCALING!!!! to be scale invariant in case of no adaptive lambda
        gradient = (
            _return_dii_gradient(
                dists_rescaled_A,
                data,
                rank_matrix_B,
                weights,
                lambd,
                period=period,
                n_jobs=n_jobs,
                cythond=cythond,
            )
            * scaling
        )
        if np.isnan(gradient).any():  # If any of the gradient elements turned to nan
            diis[i_epoch + 1] = diis[i_epoch]
            l1_penalties[i_epoch + 1] = l1_penalties[i_epoch]
            weights_list[i_epoch + 1] = weights_list[i_epoch]
            warn(
                "At least one gradient element turned to Nan, no optimization possible."
            )
            break
        else:
            # exponentially decaying lr
            if decaying_lr == True:
                lrate = l_rate * 2 ** (
                    -i_epoch / 10
                )  # every 10 epochs the learning rate will be halfed

            # Gradient Descent Clipping update (Tsuruoka 2008)
            # update rescaling weights, making sure they do not do a sign change due to learning rate step
            weights_new = weights - lrate * gradient
            for i, gam in enumerate(weights_new):
                if gam > 0:
                    weights_new[i] = max(0.0, gam - lrate * l1_penalty)
                elif gam < 0:
                    weights_new[i] = np.abs(min(0.0, gam + lrate * l1_penalty))
            weights = weights_new
            # exit the loop if all weights are 0 (e.g. l1-regularization too strong)
            if weights.any() == 0:
                diis[i_epoch + 1] = diis[i_epoch]
                l1_penalties[i_epoch + 1] = l1_penalties[i_epoch]
                weights_list[i_epoch + 1] = weights_list[i_epoch]
                warn(
                    f"The l1-regularization of "
                    + str(l1_penalty)
                    + " is too high. All features would be set to 0. No full optimization possible",
                )
                break

            # apply constrain on the weights
            if constrain:
                scaling = 1 / np.max(np.abs(weights))
            weights = scaling * weights

            # for adaptive lambda: calculate distance matrix in rescaled input
            rescaled_data_A = weights * data
            if period is not None:
                periodarray = weights * period
            dists_rescaled_A = _return_full_dist_matrix(
                rescaled_data_A, period=periodarray, cythond=cythond, n_jobs=n_jobs
            )
            lambd = (
                scaling * lambd
            )  # to make the gradient scale invariant. adaptive lambda automatically scales lambda to the features for scale invariance
            if adaptive_lambd:
                lambd = _return_optimal_lambda_from_distances(dists_rescaled_A)

            # compute DII
            diis[i_epoch + 1] = _return_dii(dists_rescaled_A, rank_matrix_B, lambd)
            l1_penalties[i_epoch + 1] = l1_penalty * np.sum(np.abs(weights))
            weights_list[i_epoch + 1] = weights
    #  if weightcheck == 1:
    #      print("The l1-regularization of ",l1_penalty," is too high. All features set to 0. No optimization possible")
    if l1_penalty == 0.0:
        return weights_list, diis, diis * 0
    else:
        return weights_list, diis, l1_penalties


@cast_ndarrays
def _optimize_dii_static_zeros(
    groundtruth_data: np.ndarray,
    data: np.ndarray,
    weights_0: np.ndarray,
    n_jobs: int,
    lambd: float = None,
    n_epochs: int = 100,
    l_rate: float = 0.1,
    constrain: bool = False,
    decaying_lr: bool = True,
    period: np.ndarray = None,
    groundtruthperiod: np.ndarray = None,
    cythond: bool = True,
):
    """Optimization where 0 weights stay 0. Used in backward eliminitaion of features
    Args:
        groundtruth_data (np.ndarray): N x D(groundtruth) array containing N datapoints in all the groundtruth features D(groundtruth)
        data (np.ndarray): N x D(input) array containing N datapoints in D input features whose weights are optimized to reproduce the groundtruth distances
        weights_0 (np.ndarray or list): D(input) initial weights for the input features. No zeros allowed here
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
        weights:
        diis:
    """
    # batch GD optimization with zeroes staying zeros - needed for return_backward_greedy_dii_elimination

    N = data.shape[0]
    D = data.shape[1]

    diis = np.ones(n_epochs + 1)  # +1: to include initial value
    weights_list = np.zeros((n_epochs + 1, D))
    scaling = 1  # if there is no constraint on rescaling of weights
    rank_matrix_B = _return_full_rank_matrix(
        groundtruth_data, period=groundtruthperiod, cythond=cythond, n_jobs=n_jobs
    )

    # initializations
    if constrain:
        scaling = 1 / np.max(np.abs(weights_0))
    weights = scaling * weights_0
    weights_list[0] = weights

    # rescale input data with the weights
    rescaled_data_A = weights * data

    # for adaptive lambda: calculate distance matrix in rescaled input
    if period is not None:
        dists_rescaled_A = _return_full_dist_matrix(
            data=rescaled_data_A,
            period=weights * period,
            cythond=cythond,
            n_jobs=n_jobs,
        )
    else:
        dists_rescaled_A = _return_full_dist_matrix(
            data=rescaled_data_A, period=period, cythond=cythond, n_jobs=n_jobs
        )

    if lambd is not None:
        lambd = (
            scaling * lambd
        )  # to make the gradient scale invariant. adaptive lambda automatically scales lambda to the features for scale invariance
        adaptive_lambd = False
    elif lambd is None:
        adaptive_lambd = True
        lambd = _return_optimal_lambda_from_distances(dists_rescaled_A)

    diis[0] = _return_dii(dists_rescaled_A, rank_matrix_B, lambd)
    lrate = l_rate  # for not expon. decaying learning rates

    for i_epoch in range(n_epochs):
        # compute gradient * SCALING!!!! to be scale invariant
        gradient = (
            _return_dii_gradient(
                dists_rescaled_A,
                data,
                rank_matrix_B,
                weights,
                lambd,
                n_jobs,
                period=period,
                cythond=cythond,
            )
            * scaling
        )
        if np.isnan(gradient).any():  # If any of the gradient elements turned to nan
            diis[i_epoch + 1] = diis[i_epoch]
            weights_list[i_epoch + 1] = weights_list[i_epoch]
            warn(
                "At least one gradient element turned to Nan, no optimization possible."
            )
            break
        else:
            # set gradient to 0 if weight was 0, so the new weight is also 0. DIFFERENT FROM REGULAR OPTIMIZATION
            gradient[weights == 0] = 0

            # exponentially decaying lr
            if decaying_lr == True:
                lrate = l_rate * 2 ** (
                    -i_epoch / 10
                )  # every 10 epochs the learning rate will be halfed

            # Gradient Descent Clipping update (Tsuruoka 2008) - only works with l1 penalty... otherwise we do not reach 0
            # update rescaling weights, making sure they do not do a sign change due to learning rate step
            weights_new = weights - lrate * gradient
            for i, gam in enumerate(weights_new):
                if gam > 0:
                    weights_new[i] = max(0.0, gam)
                elif gam < 0:
                    weights_new[i] = np.abs(min(0.0, gam))
            weights = weights_new

            # apply constrain on the weights
            if constrain:
                scaling = 1 / np.max(np.abs(weights))
            weights = scaling * weights

            # for adaptive lambda: calculate distance matrix in rescaled input
            rescaled_data_A = weights * data
            if period is not None:
                dists_rescaled_A = _return_full_dist_matrix(
                    rescaled_data_A,
                    period=period * weights,
                    cythond=cythond,
                    n_jobs=n_jobs,
                )
            else:
                dists_rescaled_A = _return_full_dist_matrix(
                    rescaled_data_A, period=period, cythond=cythond, n_jobs=n_jobs
                )

            lambd = (
                scaling * lambd
            )  # to make the gradient scale invariant. adaptive lambda automatically scales lambda to the features for scale invariance
            if adaptive_lambd:
                lambd = _return_optimal_lambda_from_distances(dists_rescaled_A)

            # compute DII
            diis[i_epoch + 1] = _return_dii(dists_rescaled_A, rank_matrix_B, lambd)
            weights_list[i_epoch + 1] = weights

    return weights_list, diis


def _refine_lasso_optimization(
    gs,
    ks,
    ls,
    l1_penalties: list,
    groundtruth_data,
    data,
    weights_0,
    n_jobs,
    lambd=None,
    n_epochs=50,
    l_rate=None,
    constrain=False,
    decaying_lr=True,
    period=None,
    groundtruthperiod=None,
    cythond=True,
    verbose=False,
):
    """Generate more lasso runs in between lasso strengths that produced non-consecutive numbers of non-zero weights

    Args:
        gs (np.ndarray): n_lassos x n_epochs x D array containing the weights during the optimization for each lasso strength in the previous optimization
        ks (np.ndarray): n_lassos x n_epochs array containing the information imbalance term of the loss in the previous optimization
        ls (np.ndarray): n_lassos x n_epochs array containing the l1-term of the loss in the previous optimization
        l1_penalties (list): list of n_lassos l1-regularization strengths tested in the previous optimization
        groundtruth_data (np.ndarray): N x D(groundtruth) array containing N datapoints in all the groundtruth features D(groundtruth)
        data (np.ndarray): N x D(input) array containing N datapoints in D input features whose weights are optimized to reproduce the groundtruth distances
        weights_0 (np.ndarray or list): D(input) initial weights for the input features. No zeros allowed here
        lambd (float): softmax scaling. If None (preferred) this chosen automatically with compute_optimial_lambda
        l_rate (float or None): if None, the learning rate is determined automatically with optimize_learning_rate
        n_epochs (int): number of epochs in each optimization cycle
        constrain (bool): if True, rescale the weights so the biggest weight = 1
        decaying_lr (bool): default: True. Apply decaying learning rate = l_rate * 2**(-i_epoch/10) - every 10 epochs the learning rate will be halfed
        period (float or np.ndarray/list): D(input) periods (input formatted to be 0-period). If not a list, the same period is assumed for all D features
        groundtruthperiod (float or np.ndarray/list): D(groundtruth) periods (groundtruth formatted to be 0-period). If not a list, the same period is assumed for all D(groundtruth) features
        cythond (bool): Flag indicating whether to use Cython-based distance computation methods.
            Should be True (default) unless you want to test the Python-based methods.
        verbose (bool): Default: False. If True, print the time it took to optimize each lasso strength.

    Returns:
        opt_l_rate (float): Learning rate, which leads to optimal unregularized (no l1-penalty) result in the specified number of epochs
        diis_list: values of the DII during optimization in n_epochs using the l_rate. Plot to ensure the optimization went well
    """
    # TODO: @wildromi typehints
    # Find where to refine the lasso and decide on new l1 penalties
    gs[np.isnan(gs)] = 0
    l0gs = np.linalg.norm(gs[:, -1, :], 0, axis=1)
    start = data.shape[1]
    refinement_needed = []
    if l0gs[0] == start:
        highest = 1 * start
        for i, l0 in enumerate(l0gs):
            if l0 < highest - 1:
                refinement_needed.append(
                    [i - 1, i, int((l0gs[i - 1] - l0) * 2)]
                )  # lower and upper limit of refinement and how many points to add
            if l0 < highest:
                highest = l0
    elif verbose:
        print("starting lasso too big")

    if not refinement_needed:
        # TODO: @wildromi, this fixes the error below, if ok, delete both TODOs
        warn("No refinement needed. Returning regular lasso optimization results.")
        return gs, ks, ls, l1_penalties

    newpenalties = []
    for i in range(len(refinement_needed)):
        newpenalties.append(
            list(
                np.linspace(
                    l1_penalties[refinement_needed[i][0]],
                    l1_penalties[refinement_needed[i][1]],
                    refinement_needed[i][2],
                )[1:-1]
            )
        )

    # Optimimize the DII for lassos that were missing, intercollate with the old data

    all_l1s = l1_penalties[
        0 : refinement_needed[0][0] + 1
    ]  # add old l1's until the first refinement # TODO: @wildromi, this throws an error sometimes?, see fix above
    all_gs = list(gs[0 : refinement_needed[0][0] + 1])
    all_ks = list(ks[0 : refinement_needed[0][0] + 1])
    all_ls = list(ls[0 : refinement_needed[0][0] + 1])

    for i, newl1 in enumerate(newpenalties):
        gs_new = np.zeros((len(newl1), n_epochs + 1, data.shape[1]))
        ks_new = np.zeros((len(newl1), n_epochs + 1))
        ls_new = np.zeros((len(newl1), n_epochs + 1))
        # do the new optimizations
        for j in range(len(newl1)):
            if verbose:
                start = time.time()

            gs_new[j], ks_new[j], ls_new[j] = _optimize_dii(
                groundtruth_data=groundtruth_data,
                data=data,
                weights_0=weights_0,
                lambd=lambd,
                n_epochs=n_epochs,
                l_rate=l_rate,
                constrain=constrain,
                l1_penalty=newl1[j],
                decaying_lr=decaying_lr,
                period=period,
                groundtruthperiod=groundtruthperiod,
                n_jobs=n_jobs,
                cythond=cythond,
            )

            if verbose:
                end = time.time()
                print(
                    "in intercollation ",
                    i + 1,
                    " of ",
                    len(newpenalties),
                    "for test l1 ",
                    j + 1,
                    " of ",
                    len(newl1),
                    ", the time was: ",
                    end - start,
                )

        # make the intercollated list of penalties
        all_l1s = all_l1s + newpenalties[i]  # add refinement
        all_gs = all_gs + list(gs_new)
        all_ks = all_ks + list(ks_new)
        all_ls = all_ls + list(ls_new)
        if i < len(newpenalties) - 1:
            all_l1s = (
                all_l1s
                + l1_penalties[
                    refinement_needed[i][1] : refinement_needed[i + 1][0] + 1
                ]
            )  # add old l1s after refinement until next refinement
            all_gs = all_gs + list(
                gs[refinement_needed[i][1] : refinement_needed[i + 1][0] + 1]
            )
            all_ks = all_ks + list(
                ks[refinement_needed[i][1] : refinement_needed[i + 1][0] + 1]
            )
            all_ls = all_ls + list(
                ls[refinement_needed[i][1] : refinement_needed[i + 1][0] + 1]
            )
        else:
            all_l1s = (
                all_l1s + l1_penalties[refinement_needed[i][1] :]
            )  # add old l1s after last refinement
            all_gs = all_gs + list(gs[refinement_needed[i][1] :])
            all_ks = all_ks + list(ks[refinement_needed[i][1] :])
            all_ls = all_ls + list(ls[refinement_needed[i][1] :])
    return np.array(all_gs), np.array(all_ks), np.array(all_ls), all_l1s


def _extract_min_diis_lasso_optimization(weights, diis, l1_penalties):
    # Which run had which number of non-zero features.
    # If the same number of non-zero features appeared in several runs, chose the lowest dii
    total_number_of_features = len(weights[0][0])
    final_weights = weights[:, -1, :]
    final_n_nonzero_features = np.linalg.norm(final_weights, 0, axis=1)
    final_diis = diis[:, -1]
    num_feats = []
    for i in range(total_number_of_features, 0, -1):
        num_feats.append(np.where(final_n_nonzero_features == i))

    min_dii_per_nfeatures = []
    for i in num_feats:
        if len(i[0]) > 0:
            minimum_imb = np.argmin(final_diis[i[0]])
            min_dii_per_nfeatures.append(i[0][minimum_imb])
        else:
            min_dii_per_nfeatures.append(np.nan)

    # Now select of the penalties, imbalances and weights by index
    # that correspond to the lowest imbalance for a certain number of non-zero features
    p_min = np.ones(len(min_dii_per_nfeatures)) * np.nan  # penalties
    dii_min = np.ones(len(min_dii_per_nfeatures)) * np.nan  # differentiable imbalances
    weights_min = (
        np.ones((len(min_dii_per_nfeatures), final_weights.shape[1])) * np.nan
    )  # weights
    for i, indexx in enumerate(min_dii_per_nfeatures):
        if np.isnan(indexx):
            pass
        else:
            p_min[i] = l1_penalties[indexx]
            dii_min[i] = final_diis[indexx]
            weights_min[i] = final_weights[indexx]
    num_nonzero_features = np.count_nonzero(np.nan_to_num(weights_min * 1, 0), axis=1)
    num_nonzero_features = np.where(
        num_nonzero_features == 0, np.nan, num_nonzero_features
    )
    return num_nonzero_features, p_min, dii_min, weights_min


def _plot_min_lasso_results(dii_min, num_nonzero_features, p_min):
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)
    dii_minmask = np.isfinite(dii_min)
    ax.plot(
        num_nonzero_features[dii_minmask],
        dii_min[dii_minmask],
        "-o",
        label="$L_1$-reg. search",
        zorder=7,
    )
    sc = ax.scatter(
        num_nonzero_features, dii_min, s=50, c=np.log(p_min, where=p_min > 0), zorder=8
    )
    cb = fig.colorbar(sc, ax=ax, orientation="vertical")
    cb.set_label(label="ln($L_1$-strength)", size="large")
    cb.ax.tick_params(labelsize="large")

    ax.set_title("Best resulting DIIs in $L_1$-regulated search", fontsize=14)
    ax.set_xlabel("Number of non-zero features", fontsize=14)
    ax.set_ylabel("DII", fontsize=14)
    ax.legend(fontsize=14)
    ax.grid(visible=True, which="major", axis="both")
    major_xticks = range(
        0, len(num_nonzero_features) + 1, len(num_nonzero_features) // 5
    )
    ax.set_xticks(major_xticks)
    ax.set_xticklabels(major_xticks, fontsize=11)
    minor_xticks = np.arange(0, len(num_nonzero_features), 1)
    ax.set_xticks(minor_xticks, minor=True)
    maxy = np.maximum(0.2, np.nanmax(dii_min))
    ax.set_ylim(0, maxy)
    major_yticks = np.around(np.arange(0, maxy + 0.01, maxy / 5), 2)
    ax.set_yticks(major_yticks)
    ax.set_yticklabels(major_yticks, fontsize=11)
    ax.grid(visible=True, which="minor", axis="x", linestyle=":", linewidth=0.5)
    plt.show()
