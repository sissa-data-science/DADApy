import multiprocessing

import numpy as np
from scipy.spatial import cKDTree as KD

from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

cores = multiprocessing.cpu_count()


# --------------------------------------------------------------------------------------


def compute_all_distances(X, n_jobs=cores):
    dists = pairwise_distances(X, Y=None, metric="euclidean", n_jobs=n_jobs)
    return dists


# --------------------------------------------------------------------------------------


def compute_NN_PBC(X, k_max, box_size=None, p=2, cutoff=np.inf):
    tree = KD(X, boxsize=box_size)
    dist, ind = tree.query(X, k=k_max + 1, p=p, distance_upper_bound=cutoff)
    return dist, ind


def from_all_distances_to_nndistances(pdist_matrix, maxk):
    dist_indices = np.asarray(np.argsort(pdist_matrix, axis=1)[:, 0 : maxk + 1])
    distances = np.asarray(np.take_along_axis(pdist_matrix, dist_indices, axis=1))
    return dist_indices, distances


def compute_nn_distances(X, maxk, metric="minkowski", p=2, period=None):
    if period is None:

        nbrs = NearestNeighbors(n_neighbors=maxk, metric=metric, p=p).fit(X)

        distances, dist_indices = nbrs.kneighbors(X)

        if metric == "hamming":
            distances *= X.shape[1]

    else:

        distances, dist_indices = compute_NN_PBC(
            X,
            maxk,
            box_size=period,
            p=p,
        )

    return distances, dist_indices


# --------------------------------------------------------------------------------------

# helper function of compute_id_diego

# TODO ADD AUTOMATIC ROOT FINDER
# def _negative_log_likelihood(self, d, mus, n1, n2, N):
#
# 	A = math.log(d)
# 	B = (n2-n1-1)*np.sum(mus**-d - 1.)
# 	C = ((n2-1)*d+1)*np.sum(np.log(mus))
#
# 	return -(A+B-C)
#
# def _argmax_lik(self, d0, mus, n1, n2, N, eps = 1.e-7):
#
# 	indx = np.nonzero(mus == 1)
# 	mus[indx] += np.finfo(self.dtype).eps
# 	max_log_lik = minimize(self._negative_log_likelihood, x0 = d0, args = (mus, n1, n2, N),
# 							method='L-BFGS-B', tol = 1.e-7, bounds = (0, 1000))
# 	return max_log_lik.x

# --------------------------------------------------------------------------------------

import re


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """sort list in human order, for both numbers and letters
    http://nedbatchelder.com/blog/200712/human_sorting.html
    """
    return [atoi(c) for c in re.split("(\d+)", text)]


def float_keys(text):
    """sort list in human order, for both numbers and letters
    http://nedbatchelder.com/blog/200712/human_sorting.html
    """
    return [atoi(c) for c in re.split(r"((?:[0-9]+\.?[0-9]*|\.[0-9]+))", text)]


# usage example:
# import glob
# datas = []
# dirr = 'my_dir/'
# files = glob.glob(dirr+'*.ext')
# files.sort(key = natural_keys)


# --------------------------------------------------------------------------------------


def stirling(n):
    return (
        np.sqrt(2 * np.pi * n) * (n / np.e) ** n * (1.0 + 1.0 / 12.0 / n)
    )  # + 1/288/n/n - 139/51840/n/n/n/)


def log_stirling(n):
    return (
        (n + 0.5) * np.log(n) - n + 0.5 * np.log(2 * np.pi) + 1.0 / 12.0 / n
    )  # -1/360/n/n/n


def binom_stirling(k, n):
    return stirling(k) / stirling(n) / stirling(k - n)


def log_binom_stirling(k, n):
    return log_stirling(k) - log_stirling(n) - log_stirling(k - n)


# --------------------------------------------------------------------------------------


def _loglik(d, mus, n1, n2, N):
    one_m_mus_d = 1.0 - mus ** (-d)
    sum = np.sum(((1 - n2 + n1) / one_m_mus_d + n2 - 1.0) * np.log(mus))
    return sum - N / d


def _argmax_loglik(dtype, d0, d1, mus, n1, n2, N, eps=1.0e-7):
    # mu can't be == 1 add some noise
    indx = np.nonzero(mus == 1)
    mus[indx] += np.finfo(dtype).eps

    l1 = _loglik(d1, mus, n1, n2, N)
    while abs(d0 - d1) > eps:
        d2 = (d0 + d1) / 2.0
        l2 = _loglik(d2, mus, n1, n2, N)
        if l2 * l1 > 0:
            d1 = d2
        else:
            d0 = d2
    d = (d0 + d1) / 2.0

    return d


def _fisher_info_scaling(id_ml, mus, n1, n2):
    N = len(mus)
    one_m_mus_d = 1.0 - mus ** (-id_ml)
    log_mu = np.log(mus)

    j0 = N / id_ml ** 2

    factor1 = np.divide(log_mu, one_m_mus_d)
    factor2 = mus ** (-id_ml)
    tmp = np.multiply(factor1 ** 2, factor2)
    j1 = (n2 - n1 - 1) * np.sum(tmp)

    return j0 + j1


# def _f(d, mu, n, N):
#     # mu can't be == 1 add some noise
#     indx = np.nonzero(mu == 1)
#     mu[indx] += np.finfo(np.float32).eps
#
#     one_m_mus_d = 1. - mu ** (-d)
#     sum = np.sum(((1 - n) / one_m_mus_d + 2. * n - 1.) * np.log(mu))
#     return sum - N / d


# --------------------------------------------------------------------------------------
# Functions used in the metric_compasisons module
# --------------------------------------------------------------------------------------


def _return_ranks(dist_indices_1, dist_indices_2, k=1):
    """Finds all the ranks according to distance 2 of the kth neighbours according to distance 1.

    Args:
        dist_indices_1 (int[:,:]): nearest neighbours according to distance1
        dist_indices_2 (int[:,:]): nearest neighbours according to distance2
        k (int): order of nearest neighbour considered for the calculation of the conditional ranks, default is 1

    Returns:
        np.array(int): ranks according to distance 2 of the first neighbour in distance 1
    """
    assert dist_indices_1.shape[0] == dist_indices_2.shape[0]

    N = dist_indices_1.shape[0]
    maxk_2 = dist_indices_2.shape[1]

    conditional_ranks = np.zeros(N)

    for i in range(N):
        idx_k_d1 = dist_indices_1[i, k]

        wr = np.where(idx_k_d1 == dist_indices_2[i])

        if len(wr[0]) == 0:
            conditional_ranks[i] = np.random.randint(maxk_2, N)
        else:
            conditional_ranks[i] = wr[0][0]

    return conditional_ranks


def _return_imbalance(dist_indices_1, dist_indices_2, k=1, dtype="mean"):
    """Compute the information imbalance between two precomputed distance measures.

    Args:
        dist_indices_1 (int[:,:]): nearest neighbours according to distance1
        dist_indices_2 (int[:,:]): nearest neighbours according to distance2
        k (int): order of nearest neighbour considered for the calculation of the imbalance, default is 1
        dtype (str): type of information imbalance computation, default is 'mean'

    Returns:
        (float): information imbalance from distance 1 to distance 2
    """
    assert dist_indices_1.shape[0] == dist_indices_2.shape[0]

    N = dist_indices_1.shape[0]

    ranks = _return_ranks(dist_indices_1, dist_indices_2, k=k)

    if dtype == "mean":
        imb = np.mean(ranks) / (N / 2.0)
    elif dtype == "log_mean":
        imb = np.log(np.mean(ranks) / (N / 2.0))
    elif dtype == "binned":
        nbins = int(round(N / k))

        Hmax = np.log(nbins)

        cs = ranks / N
        freqs, bins = np.histogram(cs, nbins, range=(0, 1.0))
        ps = freqs / N

        nonzero = np.nonzero(ps)
        ps = ps[nonzero]
        H = -np.dot(ps, np.log(ps))

        imb = H / Hmax

    else:
        raise ValueError("Choose a valid imbalance type (dtype)")
    return imb


def _return_imb_between_two(X, Xp, maxk, k, ltype="mean"):
    nbrsX = NearestNeighbors(
        n_neighbors=maxk, algorithm="auto", metric="minkowski", p=2, n_jobs=1
    ).fit(X)

    nbrsXp = NearestNeighbors(
        n_neighbors=maxk, algorithm="auto", metric="minkowski", p=2, n_jobs=1
    ).fit(Xp)

    _, dist_indices_Xp = nbrsXp.kneighbors(Xp)

    _, dist_indices_X = nbrsX.kneighbors(X)

    nXp_X = _return_imbalance(dist_indices_Xp, dist_indices_X, k=k, dtype=ltype)
    nX_Xp = _return_imbalance(dist_indices_X, dist_indices_Xp, k=k, dtype=ltype)

    return nX_Xp, nXp_X


def _return_imb_linear_comb_two_dists(
    dist_indices_Y, d1, d2, a1, maxk, k=1, ltype="mean"
):
    dX = np.sqrt((a1 ** 2 * d1 ** 2 + d2 ** 2))

    dist_indices_X = np.asarray(np.argsort(dX, axis=1)[:, 0 : maxk + 1])

    nX_Y = _return_imbalance(dist_indices_X, dist_indices_Y, k=k, dtype=ltype)
    nY_X = _return_imbalance(dist_indices_Y, dist_indices_X, k=k, dtype=ltype)

    return nX_Y, nY_X


def _return_imb_linear_comb_three_dists(
    dist_indices_Y, d1, d2, d3, a1, a2, maxk, k=1, ltype="mean"
):
    dX = np.sqrt((a1 ** 2 * d1 ** 2 + a2 ** 2 * d2 ** 2 + d3 ** 2))

    dist_indices_X = np.asarray(np.argsort(dX, axis=1)[:, 0 : maxk + 1])

    nX_Y = _return_imbalance(dist_indices_X, dist_indices_Y, k=k, dtype=ltype)
    nY_X = _return_imbalance(dist_indices_Y, dist_indices_X, k=k, dtype=ltype)

    return nX_Y, nY_X


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


def _compute_pull_variables(set1, err1, set2, err2=None):
    """Computes the pull distribution between two sets of error-affected measures.

    For each value i the pull vairable is defined as chi[i] = (set1[i]-set2[i])/sqrt(err1[i]^2+err2[i]^2).\
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
