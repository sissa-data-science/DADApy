# --------------------------------------------------------------------------------------
# Functions used in the metric_comparisons module
# --------------------------------------------------------------------------------------

import numpy as np
from sklearn.neighbors import NearestNeighbors


def _return_ranks(dist_indices_1, dist_indices_2, k=1):
    """Finds all the ranks according to distance 2 of the kth neighbours according to distance 1.

    Args:
        dist_indices_1 (np.ndarray(int)): N x maxk matrix, nearest neighbours according to distance1
        dist_indices_2 (np.ndarray(int))): N x maxk_2 matrix, nearest neighbours according to distance2
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


def _return_imbalance(dist_indices_1, dist_indices_2, k=1):
    """Compute the information imbalance between two precomputed distance measures.

    Args:
        dist_indices_1 (np.ndarray(int): nearest neighbours according to distance1
        dist_indices_2 (np.ndarray(int): nearest neighbours according to distance2
        k (int): order of nearest neighbour considered for the calculation of the imbalance, default = 1

    Returns:
        (float): information imbalance from distance 1 to distance 2

    """
    assert dist_indices_1.shape[0] == dist_indices_2.shape[0]

    N = dist_indices_1.shape[0]

    ranks = _return_ranks(dist_indices_1, dist_indices_2, k=k)

    imb = np.mean(ranks) / (N / 2.0)

    return imb
