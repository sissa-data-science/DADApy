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
import pytest
from scipy.spatial.distance import cdist

from dadapy import Clustering
from dadapy._cython import cython_clustering as cf
from dadapy._cython import cython_density as cd

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


@pytest.fixture(scope="module")
def adp_clustering():
    """Return a fitted ADP clustering object for prediction tests."""
    cl = Clustering(coordinates=X)
    cl.compute_clustering_ADP(Z=1.65)
    return cl


def test_clustering_ADP_cython():
    """Test the clustering operations work correctly."""
    cl = Clustering(coordinates=X)

    _, _ = cl.compute_clustering_ADP(Z=1.65)

    assert cl.N_clusters == 2

    assert (cl.cluster_assignment == expected_cluster_assignment).all()

    assigned_points = [point for cluster in cl.cluster_indices for point in cluster]
    assert sorted(assigned_points) == list(range(cl.N))
    assert len(assigned_points) == len(set(assigned_points))
    assert len(cl.cluster_indices) == cl.N_clusters
    assert len(cl.cluster_centers) == cl.N_clusters
    for cluster, indices in enumerate(cl.cluster_indices):
        assert np.all(cl.cluster_assignment[indices] == cluster)

    valid_halo_labels = (cl.cluster_assignment_halo == -1) | (
        (cl.cluster_assignment_halo >= 0) & (cl.cluster_assignment_halo < cl.N_clusters)
    )
    assert valid_halo_labels.all()

    expected_matrix_shape = (cl.N_clusters, cl.N_clusters)
    assert cl.log_den_bord.shape == expected_matrix_shape
    assert cl.log_den_bord_err.shape == expected_matrix_shape
    assert cl.bord_indices.shape == expected_matrix_shape


def test_clustering_adp_merges_clusters_at_higher_z():
    """A larger merging factor merges the two modes in the reference data."""
    cl = Clustering(coordinates=X)
    cl.compute_clustering_ADP(Z=3.0)

    assert cl.N_clusters == 1
    assert np.all(cl.cluster_assignment == 0)


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


@pytest.mark.parametrize("density_est", ["PAk", "kstarNN"])
def test_predict_cluster_adp_distance_inputs_are_equivalent(
    adp_clustering, density_est
):
    """Predictions are independent of the supported distance input format."""
    cl = adp_clustering

    query_indices = np.array([0, 10, 50, 60])
    X_new = X[query_indices] + 1e-8
    maxk = 20

    cross_distances = cdist(X_new, X)
    cross_dist_indices = np.argsort(cross_distances, axis=1)[:, :maxk]
    nn_distances = np.take_along_axis(cross_distances, cross_dist_indices, axis=1)

    predictions_computed = cl.predict_cluster_ADP(
        X_new, maxk=maxk, density_est=density_est
    )
    predictions_matrix = cl.predict_cluster_ADP(
        X_new,
        maxk=maxk,
        distances=cross_distances,
        density_est=density_est,
    )
    predictions_tuple = cl.predict_cluster_ADP(
        X_new,
        maxk=maxk,
        distances=(nn_distances, cross_dist_indices),
        density_est=density_est,
    )

    assert np.array_equal(predictions_computed[0], cl.cluster_assignment[query_indices])

    cluster_prediction, cluster_prediction_halo, probability, probability_halo = (
        predictions_computed
    )
    n_queries = len(X_new)
    assert cluster_prediction.shape == (n_queries,)
    assert cluster_prediction_halo.shape == (n_queries,)
    assert probability.shape == (n_queries, cl.N_clusters)
    assert probability_halo.shape == (n_queries, cl.N_clusters + 1)
    assert np.all((cluster_prediction >= 0) & (cluster_prediction < cl.N_clusters))
    assert np.all(
        (cluster_prediction_halo == -1)
        | ((cluster_prediction_halo >= 0) & (cluster_prediction_halo < cl.N_clusters))
    )
    assert np.all(np.isin(probability, [0, 1]))
    assert np.all(np.isin(probability_halo, [0, 1]))
    assert np.all(probability.sum(axis=1) == 1)
    assert np.all(probability_halo.sum(axis=1) == 1)

    for expected, matrix_result, tuple_result in zip(
        predictions_computed, predictions_matrix, predictions_tuple
    ):
        assert np.array_equal(matrix_result, expected)
        assert np.array_equal(tuple_result, expected)


def test_predict_cluster_adp_forwards_kstar_options(adp_clustering, monkeypatch):
    """Prediction forwards alpha and Bonferroni options to interpolated kstar."""
    recorded_options = {}
    compute_kstar_interp = cd._compute_kstar_interp

    def record_kstar_options(*args):
        recorded_options["alpha"] = args[3]
        recorded_options["bonferroni_deloc"] = args[7]
        recorded_options["bonferroni_loc"] = args[8]
        return compute_kstar_interp(*args)

    monkeypatch.setattr(cd, "_compute_kstar_interp", record_kstar_options)

    adp_clustering.predict_cluster_ADP(
        X[[0, 50]] + 1e-8,
        maxk=20,
        alpha=0.05,
        bonferroni_deloc=True,
        bonferroni_loc=True,
    )

    assert recorded_options == {
        "alpha": 0.05,
        "bonferroni_deloc": True,
        "bonferroni_loc": True,
    }


def test_predict_cluster_adp_rejects_invalid_density_estimator(adp_clustering):
    """Prediction reports unsupported density estimators explicitly."""
    with pytest.raises(ValueError, match="density_est"):
        adp_clustering.predict_cluster_ADP(
            X[[0]] + 1e-8,
            maxk=20,
            density_est="invalid",
        )


def test_predict_cluster_adp_requires_clustering():
    """Prediction requires ADP clustering attributes to be available."""
    cl = Clustering(coordinates=X)

    with pytest.raises(RuntimeError, match="ADP clustering"):
        cl.predict_cluster_ADP(X[[0]] + 1e-8, maxk=20)


def test_predict_cluster_adp_requires_distances_without_training_coordinates():
    """A distance-only model requires cross distances for prediction."""
    cl = Clustering(distances=cdist(X, X))
    cl.compute_clustering_ADP(Z=1.65)
    X_new = X[[0]] + 1e-8

    with pytest.raises(ValueError, match="distances must be supplied"):
        cl.predict_cluster_ADP(X_new, maxk=20)

    prediction = cl.predict_cluster_ADP(
        X_new,
        maxk=20,
        distances=cdist(X_new, X),
    )[0]
    assert prediction.shape == (1,)


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
        2.0,  # Z
        kstar,
        dist_indices,
        maxk,
        False,  # verb
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
