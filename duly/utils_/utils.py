import multiprocessing

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

cores = multiprocessing.cpu_count()


def compute_all_distances(X, n_jobs=cores):
    dists = pairwise_distances(X, Y=None, metric='euclidean', n_jobs=n_jobs)

    return dists


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

def _loglik(d, mus, n1, n2, N):
    one_m_mus_d = 1. - mus ** (-d)
    sum = np.sum(((1 - n2 + n1) / one_m_mus_d + n2 - 1.) * np.log(mus))
    return sum - N / d


def _argmax_loglik(dtype, d0, d1, mus, n1, n2, N, eps=1.e-7):
    # mu can't be == 1 add some noise
    indx = np.nonzero(mus == 1)
    mus[indx] += np.finfo(dtype).eps

    l1 = _loglik(d1, mus, n1, n2, N)
    while (abs(d0 - d1) > eps):
        d2 = (d0 + d1) / 2.
        l2 = _loglik(d2, mus, n1, n2, N)
        if l2 * l1 > 0:
            d1 = d2
        else:
            d0 = d2
    d = (d0 + d1) / 2.

    return d


def _fisher_info_scaling(id_ml, mus, n1, n2):
    N = len(mus)
    one_m_mus_d = 1. - mus ** (-id_ml)
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


def _return_ranks(dist_indices_1, dist_indices_2, maxk_2, k=1):
    assert (dist_indices_1.shape[0] == dist_indices_2.shape[0])

    N = dist_indices_1.shape[0]

    losses = np.zeros(N)

    for i in range(N):
        idx_k_d1 = dist_indices_1[i, k]

        wr = np.where(idx_k_d1 == dist_indices_2[i])

        if len(wr[0]) == 0:
            losses[i] = np.random.randint(maxk_2, N)
        else:
            losses[i] = wr[0][0]

    return losses


def _return_ranks_wdegeneracy(distances_1, distances_2):
    N = distances_1.shape[0]
    losses = np.zeros(N)

    for i in range(N):
        # find all occurrences of first neigh

        dists1 = distances_1[i]

        # nnd = np.partition(dists1, 1)[1]
        nnd = np.sort(dists1)[1]

        nn_indices = np.where(dists1 == nnd)[0]

        # print(nn_indices)
        if i in nn_indices:
            nn_indices = np.delete(nn_indices, np.where(i == nn_indices))

        losses[i] = 0
        nn = 0

        lnn_indices = len(nn_indices)
        # if lnn_indices > 1: print('found a point with {} NNs'.format(lnn_indices))

        for nn_index in nn_indices:
            nn += 1

            dists2i = distances_2[i]

            # find all elements at a given distance in the second metric
            d2_nndist = dists2i[nn_index]

            d2_nnidx = sum(dists2i <= d2_nndist) - 1

            d2_nndeg = sum(dists2i == d2_nndist) - 1

            # if d2_nndeg > 0: print('found {} points at given dist '.format(d2_nndeg))

            losses[i] += (d2_nnidx - (d2_nndeg) / 2.)

        losses[i] /= nn

    return losses


def _return_imbalance(dist_indices_1, dist_indices_2, maxk_2, k=1, dtype='mean'):
    assert (dist_indices_1.shape[0] == dist_indices_2.shape[0])

    N = dist_indices_1.shape[0]

    ranks = _return_ranks(dist_indices_1, dist_indices_2, maxk_2, k=k)

    if dtype == 'mean':
        imb = np.mean(ranks) / (N / 2.)
    elif dtype == 'log_mean':
        imb = np.log(np.mean(ranks) / (N / 2.))
    elif dtype == 'binned':
        nbins = int(round(N / k))
        # print(nbins)
        Hmax = np.log(nbins)

        cs = ranks / N
        freqs, bins = np.histogram(cs, nbins, range=(0, 1.))
        ps = freqs / N

        nonzero = np.nonzero(ps)
        ps = ps[nonzero]
        H = - np.dot(ps, np.log(ps))

        imb = H / Hmax
        # print(cs,freqs, H, imb)

    else:
        raise ValueError("Choose a valid imb type")
    return imb


def _return_imb_ij(i, j, maxk, X, k, dtype):
    X_ = X[:, [i]]

    nbrs = NearestNeighbors(n_neighbors=maxk, algorithm='auto', metric='minkowski',
                            p=2, n_jobs=1).fit(X_)

    _, dist_indices_i = nbrs.kneighbors(X_)

    X_ = X[:, [j]]

    nbrs = NearestNeighbors(n_neighbors=maxk, algorithm='auto', metric='minkowski',
                            p=2, n_jobs=1).fit(X_)

    _, dist_indices_j = nbrs.kneighbors(X_)

    nij = _return_imbalance(dist_indices_i, dist_indices_j, maxk, k=k, dtype=dtype)
    nji = _return_imbalance(dist_indices_j, dist_indices_i, maxk, k=k, dtype=dtype)

    print('computing loss with coord number ', i)
    return nij, nji


def _return_imb_with_coords(X, coords, dist_indices, maxk, k, ltype='mean'):
    X_ = X[:, coords]

    nbrs = NearestNeighbors(n_neighbors=maxk, algorithm='auto', metric='minkowski',
                            p=2, n_jobs=1).fit(X_)

    _, dist_indices_i = nbrs.kneighbors(X_)

    ni0 = _return_imbalance(dist_indices_i, dist_indices, maxk, k=k, dtype=ltype)
    n0i = _return_imbalance(dist_indices, dist_indices_i, maxk, k=k, dtype=ltype)
    print('computing loss with coords ', coords)
    return n0i, ni0


def _get_loss_between_two(X, Xp, maxk, k, ltype='mean'):
    nbrsX = NearestNeighbors(n_neighbors=maxk, algorithm='auto', metric='minkowski',
                             p=2, n_jobs=1).fit(X)

    nbrsXp = NearestNeighbors(n_neighbors=maxk, algorithm='auto', metric='minkowski',
                              p=2, n_jobs=1).fit(Xp)

    _, dist_indices_Xp = nbrsXp.kneighbors(Xp)

    _, dist_indices_X = nbrsX.kneighbors(X)

    nXp_X = _return_imbalance(dist_indices_Xp, dist_indices_X, maxk, k=k, dtype=ltype)
    nX_Xp = _return_imbalance(dist_indices_X, dist_indices_Xp, maxk, k=k, dtype=ltype)

    return nX_Xp, nXp_X


def _get_loss_linear_comb_two_dists_wdeg(dY, d1, d2, a1):
    dX = np.sqrt((a1 ** 2 * d1 ** 2 + d2 ** 2))

    nX_Y = _return_loss_wdegeneracy(dX, dY)
    nY_X = _return_loss_wdegeneracy(dY, dX)

    return nX_Y, nY_X


def _get_loss_linear_comb_three_dists_wdeg(dY, d1, d2, d3, a1, a2):
    dX = np.sqrt((a1 ** 2 * d1 ** 2 + a2 ** 2 * d2 ** 2 + d3 ** 2))

    nX_Y = _return_loss_wdegeneracy(dX, dY)
    nY_X = _return_loss_wdegeneracy(dY, dX)

    return nX_Y, nY_X


def _get_loss_linear_comb_two_dists(dist_indices_Y, d1, d2, a1, maxk, k=1,
                                    ltype='mean'):
    dX = np.sqrt((a1 ** 2 * d1 ** 2 + d2 ** 2))

    dist_indices_X = np.asarray(np.argsort(dX, axis=1)[:, 0:maxk + 1])

    nX_Y = _return_imbalance(dist_indices_X, dist_indices_Y, maxk, k=k, dtype=ltype)
    nY_X = _return_imbalance(dist_indices_Y, dist_indices_X, maxk, k=k, dtype=ltype)

    return nX_Y, nY_X


def _get_loss_linear_comb_three_dists(dist_indices_Y, d1, d2, d3, a1, a2, maxk, k=1,
                                      ltype='mean'):
    dX = np.sqrt((a1 ** 2 * d1 ** 2 + a2 ** 2 * d2 ** 2 + d3 ** 2))

    dist_indices_X = np.asarray(np.argsort(dX, axis=1)[:, 0:maxk + 1])

    nX_Y = _return_imbalance(dist_indices_X, dist_indices_Y, maxk, k=k, dtype=ltype)
    nY_X = _return_imbalance(dist_indices_Y, dist_indices_X, maxk, k=k, dtype=ltype)

    return nX_Y, nY_X


def _get_loss_between_two_one_fixed(X, dist_indices_Xp, maxk, k, ltype='mean'):
    nbrsX = NearestNeighbors(n_neighbors=maxk, algorithm='auto', metric='minkowski',
                             p=2, n_jobs=1).fit(X)

    _, dist_indices_X = nbrsX.kneighbors(X)

    nXp_X = _return_imbalance(dist_indices_Xp, dist_indices_X, maxk, k=k, dtype=ltype)
    nX_Xp = _return_imbalance(dist_indices_X, dist_indices_Xp, maxk, k=k, dtype=ltype)

    return nX_Xp, nXp_X


def _return_loss_wdegeneracy(distances_1, distances_2):
    N = distances_1.shape[0]

    ranks = _return_ranks_wdegeneracy(distances_1, distances_2)

    return (np.mean(ranks) - 1) / (N / 2. - 1)


def load_coords_and_losses(coords_file='selected_coords.txt',
                           losses_file='all_losses.npy'):
    coords = np.genfromtxt(coords_file).astype(int)

    losses = np.load(losses_file)

    return coords, losses


def _align_arrays(set1, err1, set2, err2=None):
    """Computes the constant offset between two sets of error-affected measures and returns the first array aligned to the second, shifted by such offset.

    The offset is computed by inverse-variance weighting least square linear regression of a constant law on the differences between the two sets.

    Args:
        set1 (np.array(float)): array containing the first set of values, to be aligned to set2.
        err1 (np.array(float)): array containing the statistical errors on the values set1.
        set2 (np.array(float)): array containing the reference set of values, to which set1 will be aligned.
        err2 (np.array(float), optional): array containing the statistical errors on the values set2. If not given, set2 is assumed to contain errorless measures

        Returns:
            new_set2 (np.array(float)): set1 - offset
            offset (float): constant offset between the two sets
    """

    if (err2 is None):
        assert(set1.shape == set2.shape == err1.shape)
        w = 1./np.square(err1)
    
    else:
        assert(set1.shape == set2.shape == err1.shape == err2.shape)
        w = 1./(np.square(err1)+np.square(err2))

    diffs = set1 - set2
    offset = np.average(diffs,weights=w)

    return offset, set1-offset


def _compute_pull_variables(set1, err1, set2, err2=None):
    """Computes the pull distribution between two sets of error-affected measures.

    For each value i the pull vairable is defined as chi[i] = (set1[i]-set2[i])/sqrt(err1[i]^2+err2[i]^2).\
    If err2 is not given, set2 is assumed to contain errorless measures.

    Args:
        set1 (np.array(float)): array containing the first set of values, to be aligned to set2.
        err1 (np.array(float)): array containing the statistical errors on the values set1.
        set2 (np.array(float)): array containing the reference set of values.
        err2 (np.array(float), optional): array containing the statistical errors on the values set2. If not given, set2 is assumed to contain errorless measures

        Returns:
            pull (np.array(float)): array of the pull variables
    """

    if (err2 is None):
        assert(set1.shape == set2.shape == err1.shape)
        return ( set1 - set2 ) / err1
    
    else:
        assert(set1.shape == set2.shape == err1.shape == err2.shape)
        den = np.sqrt(np.square(err1)+np.square(err2))
        return ( set1 - set2 ) / den    
