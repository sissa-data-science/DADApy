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

import numpy as np


def _return_ranks(dist_indices_1, dist_indices_2, rng, k=1):
    """Finds all the ranks according to distance 2 of the neighbours according to distance 1.
       Neighbours in distance 1 are considered up to order k.

    Args:
        dist_indices_1 (np.ndarray(int)): N x maxk matrix, nearest neighbours according to distance 1
        dist_indices_2 (np.ndarray(int))): N x maxk_2 matrix, nearest neighbours according to distance 2
        k (int): order of nearest neighbour considered for the calculation of the conditional ranks, default is 1

    Returns:
        conditional_ranks (np.ndarray(int)): N x k matrix, ranks according to distance 2 of the neighbours in distance 1

    """
    assert dist_indices_1.shape[0] == dist_indices_2.shape[0]

    N = dist_indices_1.shape[0]
    maxk_2 = dist_indices_2.shape[1]

    conditional_ranks = np.zeros((N, k))

    for i in range(N):
        idx_k_d1 = dist_indices_1[i, 1 : k + 1]

        wr = [
            np.where(idx_k_d1[k_neighbor] == dist_indices_2[i])[0]
            for k_neighbor in range(k)
        ]

        for k_neighbor in range(k):
            if len(wr[k_neighbor]) == 0:
                conditional_ranks[i, k_neighbor] = rng.integers(
                    low=maxk_2, high=N, size=1
                )
            else:
                conditional_ranks[i, k_neighbor] = wr[k_neighbor][0]

    return conditional_ranks


def _return_imbalance(dist_indices_1, dist_indices_2, rng, k=1):
    """Compute the information imbalance between two precomputed distance measures.

    Args:
        dist_indices_1 (np.ndarray(int)): nearest neighbours according to distance 1
        dist_indices_2 (np.ndarray(int)): nearest neighbours according to distance 2
        k (int): order of nearest neighbour considered for the calculation of the imbalance, default = 1

    Returns:
        (float): information imbalance from distance 1 to distance 2

    """
    assert dist_indices_1.shape[0] == dist_indices_2.shape[0]

    N = dist_indices_1.shape[0]

    ranks = _return_ranks(dist_indices_1, dist_indices_2, rng=rng, k=k)

    imb = np.mean(ranks) / (N / 2.0)

    return imb


def _return_period_present(
    period_cause,
    period_effect,
    period_conditioning,
    dim_cause,
    dim_effect,
    dim_conditioning,
    weight,
):
    """Compute the array of periods for the space (weight1 * cause, weight2 * conditioning, effect).

    Args:
        period_cause (float, np.ndarray(float)): periods of variables in 'cause' system
        period_effect (float, np.ndarray(float)): periods of variables in 'effect' system
        period_conditioning (float, np.ndarray(float)): periods of variables in 'conditioning' system
        dim_cause (int): number of variables in 'cause' system
        dim_effect (int): number of variables in 'effect' system
        dim_conditioning (int): number of variables in 'conditioning' system
        weight (float, np.ndarray(float)): single weight if 'period_conditioning' is None, array of
            shape (2,) if 'period_conditioning' is not None

    Returns:
        (np.ndarray(float)): array of shape (dim_cause + dim_effect + dim_conditioning,) with the
            concatenated periods of the three spaces, or 'None' if all the periods are None (PBCs not applied)

    """
    period_present = period_effect
    dim_present = dim_effect

    if period_conditioning is not None:
        period_present = _return_period_mixed(
            period_conditioning,
            period_present,
            dim_conditioning,
            dim_present,
            weight[1],
            1,
        )
        dim_present = period_present.shape[0]
    else:
        weight = [weight, None]

    period_present = _return_period_mixed(
        period_cause,
        period_present,
        dim_cause,
        dim_present,
        weight[0],
        1,
    )

    return period_present


def _return_period_mixed(period1, period2, dim1, dim2, weight1=1, weight2=1):
    """Compute the array of periods for a space obtained by concatenating two sets of coordinates.

    Args:
        period1 (float, np.ndarray(float)): periods of variables in first space
        period2 (float, np.ndarray(float)): periods of variables in second space
        dim1 (int): number of variables in first space
        dim2 (int): number of variables in second space
        weight1 (float): scaling weight for variables in first space
        weight2 (float): scaling weight for variables in second space

    Returns:
        (np.ndarray(float)): array of shape (dim1 + dim2,) with the concatenated periods of the
            scaled feature spaces, or 'None' if period1 and period2 are None (PBCs not applied)

    """
    if period1 is None and period2 is None:
        period_mixed = None
    elif (period1 is None and period2 is not None) or (
        period1 is not None and period2 is None
    ):
        raise ValueError(
            "'period1' is None while 'period2' is not None, but this is not supported by the current "
            + "implementation.\nIf you do not want to compute periodic distances for one of the "
            + "systems, set a period larger than the maximum variation of its features."
        )
    else:
        assert (
            isinstance(period1, (int, float))
            or (isinstance(period1, (np.ndarray, list)) and len(period1) == dim1)
        ) and (
            isinstance(period2, (int, float))
            or (isinstance(period2, (np.ndarray, list)) and len(period2) == dim2)
        ), (
            "'period1' and 'period2' must be either float scalars or numpy arrays of floats "
            + f"of shapes ({dim1},) and ({dim2},)"
        )
        if isinstance(period1, (int, float)):
            period1 = np.full(dim1, fill_value=period1, dtype=float)
        if isinstance(period2, (int, float)):
            period2 = np.full(dim2, fill_value=period2, dtype=float)
        if weight1 == 0:  # avoid period scaling when features are scaled by zero weight
            weight1 = 1
        if weight2 == 0:
            weight2 = 1
        period_mixed = np.concatenate((weight1 * period1, weight2 * period2))

    return period_mixed


def _compute_2d_grid(points_x, points_y):
    """Compute the points of the 2D grid generated by points_x and points_y.

    Args:
        points_x (list(float), np.ndarray(float)): 1D grid of Nx points along the x axis
        points_y (list(float), np.ndarray(float)): 1D grid of Ny points along the y axis
    Returns:
        (np.ndarray(float)): array of shape (Nx * Ny, 2) containing the coordinates of the points
            in the 2D grid
    """
    if (not isinstance(points_x, (list, np.ndarray))) or (
        not isinstance(points_y, (list, np.ndarray))
    ):
        raise ValueError(
            "'alphas1' and 'alphas2' must be one-dimensional lists or np.ndarrays"
        )

    X, Y = np.meshgrid(points_x, points_y)
    grid_points = np.vstack([X.ravel(), Y.ravel()]).T

    return grid_points
