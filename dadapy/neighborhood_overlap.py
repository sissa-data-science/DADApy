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

"""
The *neighborhood_overlap* module contains the *NeighborhoodOverlap* class.

Algorithms for comparing different spaces via the neighbour overlap are implemented as
methods of this class.
"""

import warnings
from collections import Counter

import numpy as np

from dadapy._cython import cython_overlap as c_ov
from dadapy._utils.metric_comparisons import _get_nn_indices
from dadapy._utils.utils import cores
from dadapy.base import Base


class NeighborhoodOverlap(Base):
    """Class for neighbour-overlap-based comparisons between metric spaces."""

    def __init__(
        self,
        coordinates=None,
        other=None,
        labels=None,
        distances=None,
        maxk=None,
        period=None,
        verbose=False,
        n_jobs=cores,
        rng_seed=42,
    ):
        """Class with methods to compare metric spaces using neighbour overlap.

        When ``other`` and/or ``labels`` are provided, methods such as
        :meth:`return_data_overlap` and :meth:`return_label_overlap` use them as
        defaults, enabling the symmetric call patterns
        ``NeighborhoodOverlap(X1, X2).return_data_overlap(k=30)`` and
        ``NeighborhoodOverlap(X, labels=y).return_label_overlap(k=5)``. Both can
        be set on the same instance.

        Args:
            coordinates (np.ndarray(float)): the data points loaded, of shape (N , dimension of embedding space)
            other (np.ndarray(float), optional): a second dataset of shape (N, D') used as
                the comparison space in :meth:`return_data_overlap`. Stored on
                ``self.X_other``.
            labels (np.ndarray, optional): labels used by :meth:`return_label_overlap`.
                Stored on ``self.labels``.
            distances (np.ndarray(float)): A matrix of dimension N x mask containing distances between points
            maxk (int): maximum number of neighbours to be considered for the calculation of distances
            period (np.array(float), optional): array containing the periodicity of each coordinate. Default is None
            verbose (bool): whether you want the code to speak or shut up
            n_jobs (int): number of cores to be used
            rng_seed (int): seed used to build ``self.rng``.
        """
        super().__init__(
            coordinates=coordinates,
            distances=distances,
            maxk=maxk,
            period=period,
            verbose=verbose,
            n_jobs=n_jobs,
            rng_seed=rng_seed,
        )
        self.X_other = other
        self.labels = labels

    def _label_imbalance_helper(self, labels, k, class_fraction):
        if k is not None:
            max_k = k
            k_per_sample = np.array([k for _ in range(len(labels))])

        k_per_class = {}
        class_count = Counter(labels)
        # potentially overwrites k_per_sample
        if class_fraction is not None:
            for label, count in class_count.items():
                class_k = int(count * class_fraction)
                k_per_class[label] = class_k
                if class_k == 0:
                    k_per_class[label] = 1
                    warnings.warn(
                        f" max_k < 1 for label {label}. max_k set to 1.\
                        Consider increasing class_fraction.",
                        stacklevel=2,
                    )
            max_k = max([k for k in k_per_class.values()])
            k_per_sample = np.array([k_per_class[label] for label in labels])

        class_weights = {label: 1 / count for label, count in class_count.items()}
        sample_weights = np.array([class_weights[label] for label in labels])

        return k_per_sample, sample_weights, max_k

    def return_label_overlap(
        self,
        labels=None,
        k=None,
        avg=True,
        coords=None,
        class_fraction=None,
        weighted=True,
    ):
        """Return the neighbour overlap between the full space and a set of labels.

        An overlap of 1 means that all neighbours of a point have the same label as the central point.

        Args:
            labels (list): the labels with respect to which the overlap is computed.
                If ``None``, falls back to ``self.labels`` set at construction time.
            k (int): the number of neighbours considered for the overlap.
            coords (array): subset of indices on which the overlap is computed.
            class_fraction (float): number of nearest neighbor considered expressed \
                as a fraction of the total number of class samples. \
                Useful when classes are imbalanced.
            weighted (bool): if True the overlap is weighted \
                inversely proportional to the class population.

        Returns:
            (float): the neighbour overlap with the class labels.
        """
        assert (
            k is not None or class_fraction is not None
        ), "k and class fraction are None. set al least one of them."
        if labels is None:
            labels = self.labels
        assert labels is not None, (
            "no labels provided: pass `labels=` or construct with "
            "`NeighborhoodOverlap(X, labels=y)`."
        )
        labels = labels.astype(int)
        k_per_sample, sample_weights, max_k = self._label_imbalance_helper(
            labels, k, class_fraction
        )

        dist_indices, max_k = _get_nn_indices(
            self.X,
            self.distances,
            self.dist_indices,
            max_k,
            self.maxk,
            self.metric,
            self.period,
            self._init_distances,
            coords=coords,
        )
        assert len(labels) == dist_indices.shape[0]

        neighbor_index = dist_indices[:, 1 : max_k + 1]
        ground_truth_labels = np.repeat(np.array([labels]).T, repeats=max_k, axis=1)
        overlaps = np.equal(np.array(labels)[neighbor_index], ground_truth_labels)

        if class_fraction is not None:
            nearest_neighbor_rank = np.arange(max_k)[np.newaxis, :]
            # should this overlap entry be discarded?
            mask = nearest_neighbor_rank >= k_per_sample[:, np.newaxis]
            # mask out the entries to be discarded
            overlaps[mask] = False

        overlaps = overlaps.sum(axis=1) / k_per_sample
        if avg and weighted:
            overlaps = np.average(overlaps, weights=sample_weights)
        elif avg:
            overlaps = np.mean(overlaps)

        return overlaps

    def return_data_overlap(
        self,
        coordinates=None,
        distances=None,
        dist_indices=None,
        k=30,
        avg=True,
        use_cython=True,
    ):
        """Return the neighbour overlap between the full space and another dataset.

        An overlap of 1 means that all neighbours of a point are the same in the two spaces.

        Args:
            coordinates (np.ndarray(float)): the data set to compare, of shape (N , dimension of embedding space).
                If ``coordinates``, ``distances`` and ``dist_indices`` are all ``None``,
                falls back to ``self.X_other`` set at construction time.
            distances (np.ndarray(float), tuple(np.ndarray(float), np.ndarray(float)) ):
                                        Distance matrix (see base class for shape explanation)
            k (int): the number of neighbours considered for the overlap

        Returns:
            (float): the neighbour overlap of the points
        """
        assert any(
            var is not None for var in [self.X, self.distances, self.dist_indices]
        ), "NeighborhoodOverlap should be initialized with a dataset."

        if coordinates is None and distances is None and dist_indices is None:
            assert self.X_other is not None, (
                "no second dataset provided: pass one of `coordinates`/`distances`/"
                "`dist_indices` or construct with `NeighborhoodOverlap(X1, X2)`."
            )
            coordinates = self.X_other

        dist_indices_base, k_base = _get_nn_indices(
            self.X,
            self.distances,
            self.dist_indices,
            k,
            self.maxk,
            self.metric,
            self.period,
            self._init_distances,
        )

        dist_indices_other, k_other = _get_nn_indices(
            coordinates,
            distances,
            dist_indices,
            k,
            self.maxk,
            self.metric,
            self.period,
            self._init_distances,
        )

        assert dist_indices_base.shape[0] == dist_indices_other.shape[0]
        k = min(k_base, k_other)
        ndata = self.N

        if use_cython:
            overlaps = c_ov._compute_data_overlap(
                ndata, k, dist_indices_base.astype(int), dist_indices_other.astype(int)
            )
        else:
            overlaps = -np.ones(ndata)
            for i in range(ndata):
                overlaps[i] = (
                    len(
                        np.intersect1d(
                            dist_indices_base[i, 1 : k + 1],
                            dist_indices_other[i, 1 : k + 1],
                        )
                    )
                    / k
                )

        if avg:
            overlaps = np.mean(overlaps)

        return overlaps
