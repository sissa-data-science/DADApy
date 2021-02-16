import multiprocessing

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

cores = multiprocessing.cpu_count()


def compute_all_distances(X, n_jobs=cores):
    dists = pairwise_distances(X, Y=None, metric='euclidean', n_jobs=n_jobs)

    return dists


# helper function of compute_id_diego
def _f(d, mu, n, N):
    # mu can't be == 1 add some noise
    indx = np.nonzero(mu == 1)
    mu[indx] += np.finfo(np.float32).eps

    one_m_mus_d = 1. - mu ** (-d)
    sum = np.sum(((1 - n) / one_m_mus_d + 2. * n - 1.) * np.log(mu))
    return sum - N / d


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


def _return_loss(dist_indices_1, dist_indices_2, maxk_2, k=1, ltype='mean'):
    assert (dist_indices_1.shape[0] == dist_indices_2.shape[0])

    N = dist_indices_1.shape[0]

    ranks = _return_ranks(dist_indices_1, dist_indices_2, maxk_2, k=k)

    if ltype == 'mean':
        loss = (np.mean(ranks) - k) / (N / 2. - k)
    elif ltype == 'log_mean':
        loss = (np.log(np.mean(ranks)) - np.log(k)) / (np.log(N / 2.) - np.log(k))
    elif ltype == 'binned':

        nbins = int(round(N / k))
        # print(nbins)
        Hmax = np.log(nbins)
        Hmin = 0.

        cs = ranks / N
        freqs, bins = np.histogram(cs, nbins, range=(0, 1.))
        ps = freqs / N

        nonzero = np.nonzero(ps)
        ps = ps[nonzero]
        H = - np.dot(ps, np.log(ps))

        loss = (H - Hmin) / (Hmax - Hmin)
        # print(cs,freqs, H, loss)


    else:
        raise ValueError("Choose a valid loss type")

    return loss


def _get_loss_with_coords(X, coords, dist_indices, maxk, k, ltype='mean'):
    X_ = X[:, coords]

    nbrs = NearestNeighbors(n_neighbors=maxk, algorithm='auto', metric='minkowski',
                            p=2, n_jobs=1).fit(X_)

    _, dist_indices_i = nbrs.kneighbors(X_)

    ni0 = _return_loss(dist_indices_i, dist_indices, maxk, k=k, ltype=ltype)
    n0i = _return_loss(dist_indices, dist_indices_i, maxk, k=k, ltype=ltype)
    print('computing loss with coords ', coords)
    return n0i, ni0


def _get_loss_between_two(X, Xp, maxk, k, ltype='mean'):
    nbrsX = NearestNeighbors(n_neighbors=maxk, algorithm='auto', metric='minkowski',
                             p=2, n_jobs=1).fit(X)

    nbrsXp = NearestNeighbors(n_neighbors=maxk, algorithm='auto', metric='minkowski',
                              p=2, n_jobs=1).fit(Xp)

    _, dist_indices_Xp = nbrsXp.kneighbors(Xp)

    _, dist_indices_X = nbrsX.kneighbors(X)

    nXp_X = _return_loss(dist_indices_Xp, dist_indices_X, maxk, k=k, ltype=ltype)
    nX_Xp = _return_loss(dist_indices_X, dist_indices_Xp, maxk, k=k, ltype=ltype)

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

    nX_Y = _return_loss(dist_indices_X, dist_indices_Y, maxk, k=k, ltype=ltype)
    nY_X = _return_loss(dist_indices_Y, dist_indices_X, maxk, k=k, ltype=ltype)

    return nX_Y, nY_X


def _get_loss_linear_comb_three_dists(dist_indices_Y, d1, d2, d3, a1, a2, maxk, k=1,
                                      ltype='mean'):
    dX = np.sqrt((a1 ** 2 * d1 ** 2 + a2 ** 2 * d2 ** 2 + d3 ** 2))

    dist_indices_X = np.asarray(np.argsort(dX, axis=1)[:, 0:maxk + 1])

    nX_Y = _return_loss(dist_indices_X, dist_indices_Y, maxk, k=k, ltype=ltype)
    nY_X = _return_loss(dist_indices_Y, dist_indices_X, maxk, k=k, ltype=ltype)

    return nX_Y, nY_X


def _get_loss_between_two_one_fixed(X, dist_indices_Xp, maxk, k, ltype='mean'):
    nbrsX = NearestNeighbors(n_neighbors=maxk, algorithm='auto', metric='minkowski',
                             p=2, n_jobs=1).fit(X)

    _, dist_indices_X = nbrsX.kneighbors(X)

    nXp_X = _return_loss(dist_indices_Xp, dist_indices_X, maxk, k=k, ltype=ltype)
    nX_Xp = _return_loss(dist_indices_X, dist_indices_Xp, maxk, k=k, ltype=ltype)

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
