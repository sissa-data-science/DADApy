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

"""Module for testing ADP clustering in its Cython implementation."""

import os

import numpy as np

from dadapy import Clustering
from dadapy._cython import cython_clustering as cf

filename = os.path.join(os.path.split(__file__)[0], "../2gaussians_in_2d.npy")

X = np.load(filename)

expected_cluster_assignment = np.array(
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    ]
)


def test_clustering_ADP_cython():
    """Test the clustering operations work correctly."""
    cl = Clustering(coordinates=X)

    _ = cl.compute_clustering_ADP(Z=1.65)

    assert cl.N_clusters == 2

    assert (cl.cluster_assignment == expected_cluster_assignment).all()


def test_clustering_adp_cython_matches_pure_python():
    """Cython and pure-python ADP implementations must produce identical results."""
    cl_cy = Clustering(coordinates=X)
    cl_cy.compute_clustering_ADP(Z=1.65)

    cl_py = Clustering(coordinates=X)
    cl_py.compute_clustering_ADP_pure_python(Z=1.65)

    assert cl_cy.N_clusters == cl_py.N_clusters
    assert np.array_equal(cl_cy.cluster_assignment, cl_py.cluster_assignment)
    assert np.array_equal(
        np.asarray(cl_cy.cluster_centers), np.asarray(cl_py.cluster_centers)
    )


def test_cython_clustering_assignment_within_valid_range():
    """cluster_assignment must lie in {-1} ∪ [0, N_clusters)."""
    cl = Clustering(coordinates=X)
    cl.compute_clustering_ADP(Z=1.65)
    ca = cl.cluster_assignment
    assert ((ca == -1) | ((ca >= 0) & (ca < cl.N_clusters))).all(), (
        f"cluster_assignment contains out-of-range values: "
        f"{ca[(ca < -1) | (ca >= cl.N_clusters)]}"
    )


def test_cython_compute_clustering_no_garbage_assignments_minimal_case():
    """Regression test for the off-by-one in the cython removed-centers loop.

    Engineered minimal scenario (N=12) where a single (i, j) pair triggers
    removal of a center. With the pre-fix off-by-one, the [center, max_neighbor]
    pair was split across two rows of `to_remove`, leaving the canonical row
    with max_neighbor = -1. The fallback in the preliminary cluster assignment
    then chained through g[-1] / cluster_init_[-1] and left point 0 unassigned,
    so `Last_cls = np.empty(...)` returned uninitialized memory for that slot.

    Setup:
        - point 1: g=100, isolated survivor center (kstar=1, nearest = point 0)
        - point 0: g=50, raw center; gets removed because point 1 (higher g)
          has it as nearest neighbor. Exactly one (i, j) pair matches.
        - points 2..11: low-g chain, none has point 0 as a neighbor, so the
          fallback's intersect1d with the buggy `to_remove[:, 0]` is the only
          way point 0 can be assigned.
    """
    N = 12
    g = np.zeros(N)
    g[1] = 100.0
    g[0] = 50.0
    for i in range(2, N):
        g[i] = 1.0 + i * 0.001
    log_den_err = np.full(N, 0.01)
    rho_c = g + 0.01
    kstar = np.array([2, 1] + [1] * (N - 2), dtype=np.int64)
    maxk = 6
    dist_indices = np.zeros((N, maxk), dtype=np.int64)
    for i in range(N):
        dist_indices[i, 0] = i
    dist_indices[0] = [0, 11, 10, 9, 8, 7]
    dist_indices[1] = [1, 0, 2, 3, 4, 5]
    for i in range(2, N):
        higher = (i + 1) if (i + 1) < N else 1
        others = [k for k in range(N) if k not in (i, higher, 0)][: maxk - 2]
        dist_indices[i] = [i, higher] + others

    out = cf._compute_clustering(
        2.0,
        False,
        kstar,
        dist_indices,
        maxk,
        False,
        log_den_err,
        rho_c,
        g,
        N,
    )
    cluster_indices, n_clusters, labels = out[0], out[1], out[2]

    assert n_clusters == 1
    assert (
        (labels >= 0) & (labels < n_clusters)
    ).all(), f"labels contain garbage values: {labels}"
    covered = sorted(p for ci in cluster_indices for p in ci)
    assert covered == list(range(N)), (
        f"cluster_indices does not cover all points; missing: "
        f"{sorted(set(range(N)) - set(covered))}"
    )
