import multiprocessing

import numpy as np
from scipy.spatial import cKDTree as KD

from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

import scipy.special as sp

cores = multiprocessing.cpu_count()


# --------------------------------------------------------------------------------------


def compute_all_distances(X, n_jobs=cores):
    dists = pairwise_distances(X, Y=None, metric="euclidean", n_jobs=n_jobs)
    return dists


# --------------------------------------------------------------------------------------


def compute_NN_PBC(X, k_max, box_size=None, p=2, cutoff=np.inf):
    tree = KD(X, boxsize=box_size)
    dist, ind = tree.query(X, k=k_max, p=p, distance_upper_bound=cutoff)
    return dist, ind


def from_all_distances_to_nndistances(pdist_matrix, maxk):
    dist_indices = np.asarray(np.argsort(pdist_matrix, axis=1)[:, 0 : maxk + 1])
    distances = np.asarray(np.take_along_axis(pdist_matrix, dist_indices, axis=1))
    return dist_indices, distances


def compute_cross_nn_distances(X_new, X, maxk, metric="euclidean", p=2, period=None):
    if period is None:

        # nbrs = NearestNeighbors(n_neighbors=maxk, metric=metric, p=p).fit(X)
        nbrs = NearestNeighbors(n_neighbors=maxk, metric=metric).fit(X)

        distances, dist_indices = nbrs.kneighbors(X_new)

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


def compute_nn_distances(X, maxk, metric="euclidean", p=2, period=None):
    distances, dist_indices = compute_cross_nn_distances(
        X, X, maxk + 1, metric=metric, p=p, period=period
    )
    return distances, dist_indices


def cast_to64(myarray):
    if myarray.dtype == "float32":
        myarray = myarray.astype("float64")
    return myarray

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
    mus[indx] += 1e-10#np.finfo(dtype).eps

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

# --------------------------------------------------------------------------------------

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

# --------------------------------------------------------------------------------------

def _beta_prior(k, n, r, a0=1, b0=1, plot=False):
    """Compute the posterior distribution of d given the input aggregates
    Since the likelihood is given by a binomial distribution, its conjugate prior is a beta distribution.
    However, the binomial is defined on the ratio of volumes and so do the beta distribution. As a
    consequence one has to change variable to have the distribution over d

    Args:
            k (nd.array(int) or int): number of points within the external shells
            n (nd.array(int)): number of points within the internal shells
            r (float): ratio between shells' radii
            a0 (float): beta distribution parameter, default =1 for flat prior
            b0 (float): prior initialiser, default =1 for flat prior
            plot (bool,default=False): plot the posterior and give a numerical estimate other than the analytical one
    Returns:
            mean_bayes (float): mean value of the posterior
            std_bayes (float): std of the posterior
    """
    from scipy.special import beta as beta_f
    from scipy.stats import beta as beta_d

    D_MAX = 100.0
    D_MIN = 0.0001

    a = a0 + n.sum()
    if isinstance(k, (np.int, int)):
        b = b0 + k * n.shape[0] - n.sum()
    else:
        b = b0 + k.sum() - n.sum()
    posterior = beta_d(a, b)

    if plot:
        import matplotlib.pyplot as plt

        def p_d(d):
            return abs(posterior.pdf(r ** d) * (r ** d) * np.log(r))

        dx = 0.1
        d_left = D_MIN
        d_right = D_MAX + dx + d_left
        d_range = np.arange(d_left, d_right, dx)
        P = np.array([p_d(di) for di in d_range]) * dx
        mask = P != 0
        elements = mask.sum()
        counter = 0
        # if less than 3 points !=0 are found, reduce the interval
        while elements < 3:
            dx /= 10
            d_range = np.arange(d_left, d_right, dx)
            P = np.array([p_d(di) for di in d_range]) * dx
            mask = P != 0
            elements = mask.sum()
            counter += 1

        # with more than 3 points !=0 we can restrict the domain and have a smooth distribution
        # I choose 1000 points but such quantity can be varied according to necessity
        ind = np.where(mask)[0]
        d_left = d_range[ind[0]] - 0.5 * dx if d_range[ind[0]] - dx > 0 else D_MIN
        d_right = d_range[ind[-1]] + 0.5 * dx
        d_range = np.linspace(d_left, d_right, 1000)
        dx = d_range[1]-d_range[0]
        P = np.array([p_d(di) for di in d_range]) * dx

        plt.plot(d_range, P)
        plt.xlabel("d")
        plt.ylabel("P(d)")
        E_d_emp = (d_range * P).sum()
        S_d_emp = np.sqrt((d_range * d_range * P).sum() - E_d_emp * E_d_emp)
        print("empirical average:\t", E_d_emp, "\nempirical std:\t\t", S_d_emp)

    E_d = (sp.digamma(a) - sp.digamma(a + b)) / np.log(r)
    S_d = np.sqrt((sp.polygamma(1, a) - sp.polygamma(1, a + b)) / np.log(r) ** 2)

    if plot:
        return E_d, S_d, d_range, P
    else:
        return E_d, S_d, None, None
