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
import multiprocessing
import warnings

import numpy as np
import scipy.special as sp
from scipy.spatial import cKDTree
from scipy.special import binom
from scipy.stats import beta as beta_d
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

cores = multiprocessing.cpu_count()


def compute_all_distances(X, n_jobs=cores, metric="euclidean"):
    """Compute the distances among all available points of the dataset X

    Args:
        X (np.ndarray): array of dimension N x D
        n_jobs (int): number of cores to use for the computation
        metric (str): metric used to compute the distances

    Returns:
        (np.ndarray(float)): N x N array containing the distances between the N points

    """

    dists = pairwise_distances(X, Y=None, n_jobs=n_jobs, metric=metric)
    return dists


def compute_NN_PBC(X, maxk, box_size=None, p=2, cutoff=np.inf):
    """Compute the neighbours of each point taking into account periodic boundaries conditions and eventual cutoff

    Args:
        X (np.ndarray): array of dimension N x D
        maxk (int): number of neighbours to save
        box_size (float, np.ndarray(float)): sizes of PBC walls. Single value is interpreted as cubic box.
        p (int): Minkowski p-norm used
        cutoff (float): set an upper bound to the distances. Over such threshold a np.inf will occur

    Returns:
        dist (np.ndarray(float)): N x maxk array containing the distances from each point to the first maxk nn
        ind (np.ndarray(int)): N x maxk array containing the indices of the neighbours of each point

    """

    tree = cKDTree(X, boxsize=box_size)
    dist, ind = tree.query(X, k=maxk, p=p, distance_upper_bound=cutoff)
    return dist, ind


def from_all_distances_to_nndistances(pdist_matrix, maxk):
    """Save the first maxk neighbours starting from the matrix of the distances

    Args:
        pdist_matrix (np.ndarray(float)): N x N matrix of distances
        maxk (int): number of neighbours to save

    Returns:
        distances (np.ndarray(float)): N x maxk matrix, distances of the neighbours of each point
        dist_indices (np.ndarray(int)): N x maxk matrix, indices of the neighbours of each point

    """

    dist_indices = np.asarray(np.argsort(pdist_matrix, axis=1)[:, 0 : maxk + 1])
    distances = np.asarray(np.take_along_axis(pdist_matrix, dist_indices, axis=1))
    return distances, dist_indices


def compute_cross_nn_distances(
    X_new, X, maxk, metric="euclidean", period=None, n_jobs=None
):
    """Compute distances, up to neighbour maxk, between points of X_new and points of X.

    The element distances[i,j] represents the distance between point i in dataset X and its j-th neighbour in dataset
    X_new, whose index is dist_indices[i,j]

    Args:
        X_new (np.array(float)): dataset from which distances are computed
        X (np.array(float)): starting dataset of points, from which distances are computed
        maxk (int): number of neighbours to save
        metric (str): metric used to compute the distances
        period (float, np.ndarray(float)): sizes of PBC walls. Single value is interpreted as cubic box.

    Returns:
        distances (np.ndarray(int)): N x maxk matrix, indices of the neighbours of each point
        dist_indices (np.ndarray(float)): N x maxk matrix, distances of the neighbours of each point

    """

    if period is None:
        nbrs = NearestNeighbors(n_neighbors=maxk, metric=metric, n_jobs=n_jobs).fit(X)

        distances, dist_indices = nbrs.kneighbors(X_new)

        # in case of hamming distance, make them integer
        if metric == "hamming":
            distances *= X.shape[1]

    else:
        if metric == "euclidean" or metric == "minkowski":
            p = 2
        elif metric == "manhattan":
            p = 1
        else:
            raise KeyError(
                "periodic distance computation is supported only for euclidean and manhattan metrics"
            )

        tree = cKDTree(X, boxsize=period)
        distances, dist_indices = tree.query(X_new, k=maxk, p=p, workers=n_jobs)

    return distances, dist_indices


def compute_nn_distances(X, maxk, metric="euclidean", period=None, n_jobs=None):
    """For each point, compute the distances from its first maxk nearest neighbours

    Args:
        X (np.ndarray): points array of dimension N x D
        maxk (int): number of neighbours to save
        metric (str): metric used to compute the distances
        period (float, np.ndarray(float)): sizes of PBC walls. Single value is interpreted as cubic box.

    Returns:
        distances (np.ndarray(int)): N x maxk matrix, indices of the neighbours of each point
        dist_indices (np.ndarray(float)): N x maxk matrix, distances of the neighbours of each point

    """

    distances, dist_indices = compute_cross_nn_distances(
        X, X, maxk + 1, metric=metric, period=period, n_jobs=n_jobs
    )

    zero_dists = np.sum(distances[:, 1:] <= 1.01 * np.finfo(np.float32).eps)
    if zero_dists > 0:
        warnings.warn(
            "There are points with neighbours at 0 distance, meaning the dataset probably has identical points.\n"
            "This can cause problems in various routines.\nWe suggest to either perform smearing of distances using\n"
            "remove_zero_dists()\n"
            "or remove identical points using\n"
            "remove_identical_points())."
        )

    return distances, dist_indices


def cast_to64(myarray):
    if myarray.dtype == "float32":
        myarray = myarray.astype("float64")
    return myarray


# --------------------------------------------------------------------------------------
# Helper functions


def _neg_loglik(dtype, d, mus, n1, n2):
    mus, n1, n2 = _filter_mus(dtype, mus, n1, n2)

    N = len(mus)
    term1 = (N - 1) * np.log(d)
    term2 = np.sum((n2 - n1 - 1) * np.log(mus**d - 1))
    term3 = -np.sum(np.log(sp.beta(n2 - n1, n1)))
    term4 = -np.sum((((n2 - 1) * d) + 1) * np.log(mus))

    return -(term1 + term2 + term3 + term4)


def _neg_dloglik_did(d, mus, n1, n2, N, eps):
    """Compute the negative derivative of the log likelihood with respect to the id."""
    one_m_mus_d = 1.0 - mus ** (-d)
    "regularize small numbers"
    one_m_mus_d[one_m_mus_d < 2 * eps] = 2 * eps

    summation = np.sum(((1 - n2 + n1) / one_m_mus_d + n2 - 1.0) * np.log(mus))
    return summation - (N - 1) / d


def _filter_mus(dtype, mus, n1, n2):
    indx = np.nonzero(mus == 1)
    mus[indx] += 10 * np.finfo(dtype).eps

    q3, q2 = np.percentile(mus, [95, 50])
    mu_max = (
        20 * (q3 - q2) + q2
    )  # very generous threshold: to accomodate fat tailed distributions
    select = (
        mus < mu_max
    )  # remove high mu values related very likely to overlapping datapoints
    mus = mus[select]

    if isinstance(n1, np.ndarray):
        n1 = n1[select]

    if isinstance(n2, np.ndarray):
        n2 = n2[select]

    return mus, n1, n2


def _argmax_loglik(dtype, d0, d1, mus, n1, n2, eps=1.0e-7):
    mus, n1, n2 = _filter_mus(dtype, mus, n1, n2)

    N = len(mus)
    l1 = _neg_dloglik_did(d1, mus, n1, n2, N, eps)
    while abs(d0 - d1) > eps:
        d2 = (d0 + d1) / 2.0
        l2 = _neg_dloglik_did(d2, mus, n1, n2, N, eps)
        if l2 * l1 > 0:
            d1 = d2
        else:
            d0 = d2
    d = (d0 + d1) / 2.0

    return d


def _fisher_info_scaling(id_ml, mus, n1, n2, eps):
    N = len(mus)
    one_m_mus_d = 1.0 - mus ** (-1.0 * id_ml)
    "regularize small numbers"
    one_m_mus_d[one_m_mus_d < eps] = eps
    log_mu = np.log(mus)

    j0 = N / id_ml**2

    factor1 = np.divide(log_mu, one_m_mus_d)
    factor2 = mus ** (-id_ml)
    tmp = np.multiply(factor1**2, factor2)
    j1 = np.sum((n2 - n1 - 1) * tmp)
    return j0 + j1


def log_binom_stirling(k, n):
    return (
        (k + 0.5) * np.log(k)
        - (n + 0.5) * np.log(n)
        - (k - n + 0.5) * np.log(k - n)
        - 0.5 * np.log(2 * np.pi)
    )


def binomial_loglik(d, k, n, r):
    if isinstance(k, np.ndarray):
        pk = np.histogram(k, bins=np.arange(-0.5, k.max() + 1.5))[0]
        pk = pk / pk.sum()
    else:
        pk = np.ones(k + 1)
        k = [k]
    log_binom = np.log(binom(k, n))
    if np.any(log_binom == np.inf):
        mask = np.where(log_binom == np.inf)[0]
        log_binom[mask] = log_binom_stirling(k[mask], n[mask])
    return -np.sum(
        n * d * np.log(r)
        + (k - n) * np.log(1.0 - r**d)
        + log_binom
        + np.log([pk[ki] for ki in k])
    )


def _compute_binomial_cramerrao(d, k, r, n):
    """Calculate the Cramer Rao lower bound for the variance associated with the binomial estimator

    Args:
        d (float): intrinsic dimension
        k (float, np.ndarray(float)): number of neighbours within the external shell
        r (float): ratio among internal and external radii
        n (int): number of points of the dataset

    Returns:
        CramerRao lower bound (float)
    """
    if isinstance(k, np.ndarray):
        k = k.mean()

    return (r ** (-d) - 1) / (k * n * np.log(r) ** 2)


# --------------------------------------------------------------------------------------
# Others
# --------------------------------------------------------------------------------------


def _align_arrays(set1, err1, set2, err2=None):
    """Computes the constant offset between two sets of error-affected measures and returns the first array aligned to
    the second, shifted by such offset.

    The offset is computed by inverse-variance weighting least square linear regression of a constant law on the
    differences between the two sets.

    Args:
        set1 (np.array(float)): array containing the first set of values, to be aligned to set2.
        err1 (np.array(float)): array containing the statistical errors on the values set1.
        set2 (np.array(float)): array containing the reference set of values, to which set1 will be aligned.
        err2 (np.array(float), optional): array containing the statistical errors on the values set2. If not given,
                                          set2 is assumed to contain errorless measures

        Returns:
            new_set2 (np.array(float)): set1 - offset
            offset (float): constant offset between the two sets
    """

    if err2 is None:
        assert set1.shape == set2.shape == err1.shape
        w = 1.0 / np.square(err1)

    else:
        assert set1.shape == set2.shape == err1.shape == err2.shape
        w = 1.0 / (np.square(err1) + np.square(err2))

    diffs = set1 - set2
    offset = np.average(diffs, weights=w)

    return offset, set1 - offset


# --------------------------------------------------------------------------------------


def _compute_pull_variables(set1, err1, set2, err2=None):
    """Computes the pull distribution between two sets of error-affected measures.

    For each value i the pull variable is defined as chi[i] = (set1[i]-set2[i])/sqrt(err1[i]^2+err2[i]^2).\
    If err2 is not given, set2 is assumed to contain errorless measures.

    Args:
        set1 (np.array(float)): array containing the first set of values, to be aligned to set2.
        err1 (np.array(float)): array containing the statistical errors on the values set1.
        set2 (np.array(float)): array containing the reference set of values.
        err2 (np.array(float), optional): array containing the statistical errors on the values set2. If not given,
        set2 is assumed to contain errorless measures

        Returns:
            pull (np.array(float)): array of the pull variables
    """

    if err2 is None:
        assert set1.shape == set2.shape == err1.shape
        return (set1 - set2) / err1

    else:
        assert set1.shape == set2.shape == err1.shape == err2.shape
        den = np.sqrt(np.square(err1) + np.square(err2))
        return (set1 - set2) / den


def _beta_prior(k, n, r, a0=1, b0=1, posterior_profile=False):
    """Compute the posterior distribution of d given the input aggregates
    Since the likelihood is given by a binomial distribution, its conjugate prior is a beta distribution.
    However, the binomial is defined on the ratio of volumes and so do the beta distribution. As a
    consequence one has to change variable to have the distribution over d

    Args:
        k (nd.array(int) or int): number of points within the external shells
        n (nd.array(int)): number of points within the internal shells
        r (float): ratio between shells' radii
        a0 (float): beta distribution parameter, default =1 for flat prior
        b0 (float): prior initializer, default =1 for flat prior
        plot (bool,default=False): plot the posterior and give a numerical estimate other than the analytical one
    Returns:
        mean_bayes (float): mean value of the posterior
        std_bayes (float): std of the posterior
        d_range (np.ndarray(float), optional): domain of the posterior (if plot==True)
        P (np.ndarray(float), optional): values of the posterior on the domain d_range (if plot==True)
    """
    D_MAX = 100.0
    D_MIN = 0.0001

    a = a0 + n.sum()
    if isinstance(k, (np.int64, int, float)):
        b = b0 + k * n.shape[0] - n.sum()
    else:
        b = b0 + k.sum() - n.sum()
    posterior = beta_d(a, b)

    if posterior_profile:
        import matplotlib.pyplot as plt

        def p_d(d):
            return abs(posterior.pdf(r**d) * (r**d) * np.log(r))

        dx = 0.1
        d_left = D_MIN
        d_right = D_MAX + dx + d_left
        d_range = np.arange(d_left, d_right, dx)
        P = np.array([p_d(di) for di in d_range]) * dx
        mask = P != 0
        elements = mask.sum()
        # if less than 3 points !=0 are found, reduce the interval
        while elements < 3:
            dx /= 10
            d_range = np.arange(d_left, d_right, dx)
            P = np.array([p_d(di) for di in d_range]) * dx
            mask = P != 0
            elements = mask.sum()

        # with more than 3 points !=0 we can restrict the domain and have a smooth distribution
        # I choose 1000 points but such quantity can be varied according to necessity
        ind = np.where(mask)[0]
        d_left = d_range[ind[0]] - 0.5 * dx if d_range[ind[0]] - dx > 0 else D_MIN
        d_right = d_range[ind[-1]] + 0.5 * dx
        d_range = np.linspace(d_left, d_right, 1000)
        dx = d_range[1] - d_range[0]
        P = np.array([p_d(di) for di in d_range]) * dx

        plt.plot(d_range, P)
        plt.xlabel("d")
        plt.ylabel("P(d)")
        E_d_emp = (d_range * P).sum()
        S_d_emp = np.sqrt((d_range * d_range * P).sum() - E_d_emp * E_d_emp)
        print("empirical average:\t", E_d_emp, "\nempirical std:\t\t", S_d_emp)

    E_d = (sp.digamma(a) - sp.digamma(a + b)) / np.log(r)
    S_d = np.sqrt((sp.polygamma(1, a) - sp.polygamma(1, a + b)) / np.log(r) ** 2)

    if posterior_profile:
        return E_d, S_d, d_range, P
    else:
        return E_d, S_d, None, None
