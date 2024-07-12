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
The *metric_comparisons* module contains the *MetricComparisons* class.

Algorithms for comparing different spaces are implemented as methods of this class.
"""

import multiprocessing
import warnings
from collections import Counter

import numpy as np
from joblib import Parallel, delayed

from dadapy._cython import cython_overlap as c_ov
from dadapy._utils.metric_comparisons import (
    _compute_2d_grid,
    _return_imbalance,
    _return_period_mixed,
    _return_period_present,
)
from dadapy._utils.utils import compute_nn_distances
from dadapy.base import Base

cores = multiprocessing.cpu_count()


class MetricComparisons(Base):
    """Class for the metric comparisons."""

    def __init__(
        self,
        coordinates=None,
        distances=None,
        maxk=None,
        period=None,
        verbose=False,
        n_jobs=cores,
    ):
        """Class containing several methods to compare metric spaces obtained using subsets of the data features.

        Using these methods one can assess whether two spaces are equivalent, completely independent, or whether one
        space is more informative than the other.

        Args:
            coordinates (np.ndarray(float)): the data points loaded, of shape (N , dimension of embedding space)
            distances (np.ndarray(float)): A matrix of dimension N x mask containing distances between points
            maxk (int): maximum number of neighbours to be considered for the calculation of distances
            period (np.array(float), optional): array containing the periodicity of each coordinate. Default is None
            verbose (bool): whether you want the code to speak or shut up
            n_jobs (int): number of cores to be used
        """
        super().__init__(
            coordinates=coordinates,
            distances=distances,
            maxk=maxk,
            period=period,
            verbose=verbose,
            n_jobs=n_jobs,
        )

    def return_information_imbalace(
        self, coordinates, k=1, subset_size=2000, repeats=None, avg=True
    ):
        """Return the imbalance with another dataset X.

        Args:
            coordinates (np.ndarray(float)): the coordinates of the othe dataset (N , dimension of embedding space).
            k (int): order of nearest neighbour considered for the calculation of the imbalance, default is 1,
            subset_size (int): size of the subsets on which the information imbalance is computed.
            repeats (int): the number of repetitions for the information imbalance calculation.
        Returns:
            (np.array, np.array): the information imbalances their standard error
        """
        assert self.X is not None, "information imbalance requires coordinate matrix."
        assert (
            self.X.shape[0] == coordinates.shape[0]
        ), "the two datasets must have the same number of samples"

        if repeats is None:
            repeats = self.N // subset_size

        if self.N <= subset_size:
            warnings.warn(
                "Subset size greater than the dataset size. \
                Computing information imbalance once on the entire dataset.",
                stacklevel=2,
            )
            repeats = 1
            subset_size = self.N
        elif repeats > self.N // subset_size:
            warnings.warn(
                "repeats * subset_size > dataset size. \
                setting repeats = dataset_size // subset_size.",
                stacklevel=2,
            )
            repeats = self.N // subset_size

        # subsets is a list of arrays. Each array contained the indices of points belonging to
        # the subsets
        subsets = [np.arange(self.N)]
        if repeats > 1:
            # shuffling the integers from 0 to self.N -1
            indices = self.rng.choice(self.N, self.N, replace=False)
            # splitting the indices array into 'repeats'
            subsets = np.array_split(indices, repeats)
            if len(subsets[-1]) != len(subsets[-2]):
                # all groups should have the same size
                subsets = subsets[:-1]
                repeats -= 1

        imb_ij = np.zeros(repeats)
        imb_ji = np.zeros(repeats)
        for i, idx in enumerate(subsets):
            x_base = self.X[idx]
            x_other = coordinates[idx]

            dist_indices_base, _ = self._get_nn_indices(
                x_base, None, None, subset_size - 1, force_computation=True
            )
            dist_indices_other, _ = self._get_nn_indices(
                x_other, None, None, subset_size - 1, force_computation=True
            )

            assert dist_indices_base.shape[0] == dist_indices_other.shape[0]

            imb_ij[i] = _return_imbalance(
                dist_indices_base, dist_indices_other, self.rng, k=k
            )

            imb_ji[i] = _return_imbalance(
                dist_indices_other, dist_indices_base, self.rng, k=k
            )

        if avg:
            if repeats == 1:
                return np.array([imb_ij[0], 0]), np.array([imb_ji[0], 0])
            mean_ij, err_ij = (
                np.mean(imb_ij),
                np.std(imb_ij, ddof=1) / repeats**0.5,
            )
            mean_ji, err_ji = (
                np.mean(imb_ji),
                np.std(imb_ji, ddof=1) / repeats**0.5,
            )
            return np.array([mean_ij, err_ij]), np.array([mean_ji, err_ji])

        return imb_ij, imb_ji

    def return_inf_imb_two_selected_coords(self, coords1, coords2, k=1):
        """Return the imbalances between distances taken as the i and the j component of the coordinate matrix X.

        Args:
            coords1 (list(int)): components for the first distance
            coords2 (list(int)): components for the second distance
            k (int): order of nearest neighbour considered for the calculation of the imbalance, default is 1

        Returns:
            (float, float): the information imbalance from distance i to distance j and vice versa
        """
        X_ = self.X[:, coords1]
        _, dist_indices_i = compute_nn_distances(
            X_, self.maxk, self.metric, self.period
        )

        X_ = self.X[:, coords2]
        _, dist_indices_j = compute_nn_distances(
            X_, self.maxk, self.metric, self.period
        )

        imb_ij = _return_imbalance(dist_indices_i, dist_indices_j, self.rng, k=k)
        imb_ji = _return_imbalance(dist_indices_j, dist_indices_i, self.rng, k=k)

        return imb_ij, imb_ji

    def return_inf_imb_matrix_of_coords(self, k=1):
        """Compute the information imbalances between all pairs of single features of the data.

        Args:
            k (int): number of neighbours considered in the computation of the imbalances

        Returns:
            n_mat (np.array(float)): a DxD matrix containing all the information imbalances
        """
        assert self.X is not None

        ncoords = self.dims

        n_mat = np.zeros((ncoords, ncoords))

        if self.verb:
            print(
                "computing imbalances with coord number on {} processors".format(
                    self.n_jobs
                )
            )

        nmats = Parallel(n_jobs=self.n_jobs)(
            delayed(self.return_inf_imb_two_selected_coords)([i], [j], k)
            for i in range(ncoords)
            for j in range(i)
        )

        indices = [(i, j) for i in range(ncoords) for j in range(i)]

        for idx, n in zip(indices, nmats):
            n_mat[idx[0], idx[1]] = n[0]
            n_mat[idx[1], idx[0]] = n[1]

        return n_mat

    def return_inf_imb_full_all_coords(self, k=1):
        """Compute the information imbalances between the 'full' space and each one of its D features.

        Args:
            k (int): number of neighbours considered in the computation of the imbalances

        Returns:
            (np.array(float)): a 2xD matrix containing the information imbalances between
            the original space and each of its D features.

        """
        assert self.X is not None

        ncoords = self.X.shape[1]

        coord_list = [[i] for i in range(ncoords)]
        imbalances = self.return_inf_imb_full_selected_coords(coord_list, k=k)

        return imbalances

    def return_inf_imb_full_selected_coords(self, coord_list, k=1):
        """Compute the information imbalances between the 'full' space and a selection of features.

        Args:
            coord_list (list(list(int))): a list of the type [[1, 2], [8, 3, 5], ...] where each
                sub-list defines a set of coordinates for which the information imbalance should be
                computed.
            k (int): number of neighbours considered in the computation of the imbalances

        Returns:
            (np.array(float)): a 2xL matrix containing the information imbalances between
            the original space and each one of the L subspaces defined in coord_list

        """
        assert self.X is not None
        if self.distances is None:
            self.compute_distances()

        print("total number of computations is: ", len(coord_list))

        imbalances = self.return_inf_imb_target_selected_coords(
            self.dist_indices, coord_list, k=k
        )

        return imbalances

    def return_inf_imb_target_all_coords(self, target_ranks, k=1):
        """Compute the information imbalances between the 'target' space and a all single feature spaces in X.

        Args:
            target_ranks (np.array(int)): an array containing the ranks in the target space
            k (int): number of neighbours considered in the computation of the imbalances

        Returns:
            (np.array(float)): a 2xL matrix containing the information imbalances between
            the target space and each one of the L subspaces defined in coord_list

        """
        assert self.X is not None

        ncoords = self.dims

        coord_list = [[i] for i in range(ncoords)]
        imbalances = self.return_inf_imb_target_selected_coords(
            target_ranks, coord_list, k=k
        )

        return imbalances

    def return_inf_imb_target_selected_coords(self, target_ranks, coord_list, k=1):
        """Compute the information imbalances between the 'target' space and a selection of features.

        Args:
            target_ranks (np.ndarray(int)): an array containing the ranks in the target space, could be e.g.
                the nearest neighbor ranks for a different set of variables on the same data points.
            coord_list (list(list(int))): a list of the type [[1, 2], [8, 3, 5], ...] where each
                sub-list defines a set of coordinates for which the information imbalance should be
                computed.
            k (int): number of neighbours considered in the computation of the imbalances

        Returns:
            (np.array(float)): a 2xL matrix containing the information imbalances between
            the target space and each one of the L subspaces defined in coord_list

        """
        assert self.X is not None
        assert target_ranks.shape[0] == self.X.shape[0]

        print("total number of computations is: ", len(coord_list))

        if self.verb:
            print(
                "computing loss with coord number on {} processors".format(self.n_jobs)
            )

        n1s_n2s = Parallel(n_jobs=self.n_jobs)(
            delayed(self._return_imb_with_coords)(self.X, coords, target_ranks, k)
            for coords in coord_list
        )

        return np.array(n1s_n2s).T

    def _return_imb_with_coords(self, X, coords, dist_indices, k):
        """Return the imbalances between a 'full' distance and a distance built using a subset of coordinates.

        Args:
            X: coordinate matrix
            coords: subset of coordinates to be used when building the alternative distance
            dist_indices (int[:,:]): nearest neighbours according to full distance
            k (int): number of neighbours considered in the computation of the imbalances, default is 1

        Returns:
            (float, float): the information imbalance from 'full' to 'alternative' and vice versa
        """
        X_ = X[:, coords]

        if self.period is not None:
            if isinstance(self.period, np.ndarray) and self.period.shape == (
                self.dims,
            ):
                self.period = self.period
            elif isinstance(self.period, (int, float)):
                self.period = np.full((self.dims), fill_value=self.period, dtype=float)
            else:
                raise ValueError(
                    f"'period' must be either a float scalar or a numpy array of floats of shape ({self.dims},)"
                )
            period_ = self.period[coords]
        else:
            period_ = self.period

        _, dist_indices_coords = compute_nn_distances(
            X_, self.maxk, self.metric, period_, n_jobs=self.n_jobs
        )

        imb_coords_full = _return_imbalance(
            dist_indices_coords, dist_indices, self.rng, k=k
        )
        imb_full_coords = _return_imbalance(
            dist_indices, dist_indices_coords, self.rng, k=k
        )

        return imb_full_coords, imb_coords_full

    def greedy_feature_selection_full(self, n_coords, k=1, n_best=10, symm=True):
        """Greedy selection of the set of features which is most informative about full distance measure.

           Using the n-best single features describing the full feature space, one more of all other features
           is added combinatorically to make a candidate pool of duplets. Then, using the n-best duplets describing
           the full space, one more of all other features is added to make a candidate pool of triplets, etc.
           This procedure is done until including the desired number of features (n_coords) is reached.

        Args:
            n_coords: number of coodinates after which the algorithm is stopped
            k (int): number of neighbours considered in the computation of the imbalances
            n_best (int): the n_best tuples are chosen in each iteration to combinatorically add one variable and
                calculate the imbalance until n_coords is reached
            symm (bool): whether to use the symmetrised information imbalance

        Returns:
            best_tuples (list(list(int))): best coordinates selected at each iteration
            best_imbalances (np.ndarray(float,float)): imbalances (full-->coords, coords-->full) computed at each
                iteration, belonging to the best tuple
            all_imbalances (list(list(list(int)))): all imbalances (full-->coords, coords-->full), computed
                at each iteration, belonging all greedy tuples
        """
        print("taking full space as the target representation")
        assert self.X is not None
        if self.distances is None:
            self.compute_distances()

        (
            best_tuples,
            best_imbalances,
            all_imbalances,
        ) = self.greedy_feature_selection_target(
            self.dist_indices, n_coords, k, n_best, symm
        )

        return best_tuples, best_imbalances, all_imbalances

    def greedy_feature_selection_target(
        self, target_ranks, n_coords, k, n_best, symm=True
    ):
        """Greedy selection of the set of features which is most informative about a target distance.

           Using the n-best single features describing the target_ranks, one more of all other features is added
           combinatorically to make a candidate pool of duplets. Then, using the n-best duplets describing the
           target_ranks, one more of all other features is added to make a candidate pool of triplets, etc.
           This procedure is done until including the desired number of variables (n_coords) is reached.

        Args:
            target_ranks (np.ndarray(int)): an array containing the ranks in the target space, could be e.g.
                the nearest neighbor ranks for a different set of variables on the same data points.
            n_coords: number of coodinates after which the algorithm is stopped
            k (int): number of neighbours considered in the computation of the imbalances
            n_best (int): the n_best tuples are chosen in each iteration to combinatorically add one variable
                and calculate the imbalance until n_coords is reached
            symm (bool): whether to use the symmetrised information imbalance

        Returns:
            best_tuples (list(list(int))): best coordinates selected at each iteration
            best_imbalances (np.ndarray(float,float)): imbalances (full-->coords, coords-->full) computed
                at each iteration, belonging to the best tuple
            all_imbalances (list(list(list(int)))): all imbalances (full-->coords, coords-->full), computed
                at each iteration, belonging all greedy tuples
        """
        assert self.X is not None

        dims = self.dims  # number of features / variables

        imbalances = self.return_inf_imb_target_all_coords(target_ranks, k=k)

        if symm:
            proj = np.dot(imbalances.T, np.array([np.sqrt(0.5), np.sqrt(0.5)]))
            selected_coords = np.argsort(proj)[0:n_best]
        else:
            selected_coords = np.argsort(imbalances[1])[0:n_best]

        selected_coords = [
            selected_coords[i : i + 1] for i in range(0, len(selected_coords))
        ]
        best_one = selected_coords[0]
        best_tuples = [[int(best_one)]]  # start with the best 1-tuple
        best_imbalances = [
            [
                round(float(imbalances[0][best_one]), 3),
                round(float(imbalances[1][best_one]), 3),
            ]
        ]
        all_imbalances = [
            [
                [round(float(num1), 3) for num1 in imbalances[0]],
                [round(float(num0), 3) for num0 in imbalances[1]],
            ]
        ]

        if self.verb:
            print("best single variable selected: ", best_one)

        all_single_coords = list(np.arange(dims).astype(int))

        while len(best_tuples) < n_coords:
            c_list = []
            for i in selected_coords:
                for j in all_single_coords:
                    if j not in i:
                        ii = list(i)
                        ii.append(j)
                        c_list.append(ii)
            coord_list = [
                list(e) for e in set(frozenset(d) for d in c_list)
            ]  # make sure no tuples are doubled

            imbalances_ = self.return_inf_imb_target_selected_coords(
                target_ranks, coord_list, k=k
            )

            if symm:
                proj = np.dot(imbalances_.T, np.array([np.sqrt(0.5), np.sqrt(0.5)]))
                to_select = np.argsort(proj)[0:n_best]
            else:
                to_select = np.argsort(imbalances_[1])[0:n_best]

            best_ind = to_select[0]
            best_tuples.append(coord_list[best_ind])  # append the best n-plet to list
            best_imbalances.append(
                [round(imbalances_[0][best_ind], 3), round(imbalances_[1][best_ind], 3)]
            )
            all_imbalances.append(
                [
                    [round(num0, 3) for num0 in imbalances_[0]],
                    [round(num1, 3) for num1 in imbalances_[1]],
                ]
            )
            selected_coords = np.array(coord_list)[to_select]

        return best_tuples, np.array(best_imbalances), all_imbalances

    def return_inf_imb_full_all_dplets(self, d, k=1):
        """Compute the information imbalances between the full space and all possible combinations of d coordinates.

        Args:
            d (int): target order considered (e.g., d = 2 will compute all couples of coordinates)
            k (int): number of neighbours considered in the computation of the imbalances

        Returns:
            coord_list: list of the set of coordinates for which imbalances are computed
            imbalances: the correspinding couples of information imbalances
        """
        assert self.X is not None
        if self.distances is None:
            self.compute_distances()

        coord_list, imbalances = self.return_inf_imb_target_all_dplets(
            self.dist_indices, d, k
        )

        return coord_list, imbalances

    def return_inf_imb_target_all_dplets(self, target_ranks, d, k=1):
        """Compute the information imbalances between a target distance and all combinations of d coordinates of X.

        Args:
            target_ranks (np.array(int)): an array containing the ranks in the target space
            d (int): target order considered (e.g., d = 2 will compute all couples of coordinates)
            k (int): number of neighbours considered in the computation of the imbalances

        Returns:
            coord_list: list of the set of coordinates for which imbalances are computed
            imbalances: the correspinding couples of information imbalances
        """
        assert self.X is not None
        import itertools

        print(
            "WARNING:  computational cost grows combinatorially! Don't forget to save the results."
        )

        if self.verb:
            print("computing loss between all {}-plets and the target label".format(d))

        D = self.X.shape[1]

        all_coords = list(np.arange(D).astype(int))

        coord_list = list(itertools.combinations(all_coords, d))

        imbalances = self.return_inf_imb_target_selected_coords(
            target_ranks, coord_list, k=k
        )

        return np.array(coord_list), np.array(imbalances)

    def _get_nn_indices(
        self,
        coordinates,
        distances,
        dist_indices,
        k,
        coords=None,
        force_computation=False,
    ):
        if force_computation:
            _, dist_indices = compute_nn_distances(
                coordinates, k, self.metric, self.period
            )
            return dist_indices, k

        if coords is not None:
            assert (
                coordinates is not None
            ), "when coords is not None the coordinate matrix \
                coordinates must be defined."
            X_ = coordinates[:, coords]
            _, dist_indices = compute_nn_distances(X_, k)
            return dist_indices, k

        if k > self.maxk:
            if dist_indices is None and distances is not None:
                # if we are given only a distance matrix without indices we expect it to be in square form
                assert distances.shape[0] == distances.shape[1]
                _, dist_indices, _, _ = self._init_distances(distances, k)
                return dist_indices, k
            elif coordinates is not None:
                # if coordinates are available and k > maxk distances should be recomputed
                # and nearest neighbors idenitified up to k.
                _, dist_indices = compute_nn_distances(
                    coordinates, k, self.metric, self.period
                )
                return dist_indices, k
            else:
                # we must set k=self.maxk and continue the compuation
                warnings.warn(
                    f"Chosen k = {k} is greater than max available number of\
                    nearest neighbors = {self.maxk}. Setting k = {self.maxk}",
                    stacklevel=2,
                )
                k = self.maxk

        if dist_indices is not None:
            # if nearest neighbors are available (up to maxk) return them
            return dist_indices, k

        elif distances is not None:
            # otherwise if distance matrix in square form is available find the first k nearest neighbors
            _, dist_indices, _, _ = self._init_distances(distances, k)
            return dist_indices, k
        else:
            # otherwise compute distances and nearest neighbors up to k.
            _, dist_indices = compute_nn_distances(
                coordinates, k, self.metric, self.period
            )
            return dist_indices, k

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
        self, labels, k=None, avg=True, coords=None, class_fraction=None, weighted=True
    ):
        """Return the neighbour overlap between the full space and a set of labels.

        An overlap of 1 means that all neighbours of a point have the same label as the central point.

        Args:
            labels (list): the labels with respect to which the overlap is computed.
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
        labels = labels.astype(int)
        k_per_sample, sample_weights, max_k = self._label_imbalance_helper(
            labels, k, class_fraction
        )

        dist_indices, max_k = self._get_nn_indices(
            self.X, self.distances, self.dist_indices, max_k, coords
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
            coordinates (np.ndarray(float)): the data set to compare, of shape (N , dimension of embedding space)
            distances (np.ndarray(float), tuple(np.ndarray(float), np.ndarray(float)) ):
                                        Distance matrix (see base class for shape explanation)
            k (int): the number of neighbours considered for the overlap

        Returns:
            (float): the neighbour overlap of the points
        """
        assert any(
            var is not None for var in [self.X, self.distances, self.dist_indices]
        ), "MetricComparisons should be initialized with a dataset."

        assert any(
            var is not None for var in [coordinates, distances, dist_indices]
        ), "The overlap with data requires a second dataset. \
            Provide at least one of coordinates, distances, dist_indices."

        dist_indices_base, k_base = self._get_nn_indices(
            self.X, self.distances, self.dist_indices, k
        )

        dist_indices_other, k_other = self._get_nn_indices(
            coordinates, distances, dist_indices, k
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

    def return_label_overlap_coords(self, labels, coords, k=30):
        """Return the neighbour overlap between a selection of coordinates and a set of labels.

        An overlap of 1 means that all neighbours of a point have the same label as the central point.

        Args:
            labels (np.ndarray): the labels with respect to which the overlap is computed
            coords (list(int)): a list of coordinates to consider for the distance computation
            k (int): the number of neighbours considered for the overlap

        Returns:
            (float): the neighbour overlap of the points
        """
        raise AssertionError(
            """This function is outdated and will be removed in a future version of the package. \
        Use "return_label_overlap" instead."""
        )

    def return_overlap_coords(self, coords1, coords2, k=30):
        """Return the neighbour overlap between two subspaces defined by two sets of coordinates.

        An overlap of 1 means that in the two subspaces all points have an identical neighbourhood.

        Args:
            coords1 (list(int)): the list of coordinates defining the first subspace
            coords2 (list(int)): the list of coordinates defining the second subspace
            k (int): the number of neighbours considered for the overlap

        Returns:
            (float): the neighbour overlap of the two subspaces
        """
        raise AssertionError(
            """This function is a wrong implementation of the overlap between two \
            sets of coordinates and will be removed in a future version of the package. \
            Use "return_data_overlap" instead."""
        )

    def return_label_overlap_selected_coords(self, labels, coord_list, k=30):
        """Return a list of neighbour overlaps computed on a list of selected coordinates.

        An overlap of 1 means that all neighbours of a point have the same label as the central point.

        Args:
            labels (np.ndarray): the labels with respect to which the overlap is computed
            coord_list (list(list(int))): a list of lists, with each sublist representing a set of coordinates
            k: the number of neighbours considered for the overlap

        Returns:
            (list(float)): a list of neighbour overlaps of the points
        """
        raise AssertionError(
            """This function is outdated and will be removed in a future version of the package. \
        Use "return_label_overlap" instead."""
        )

    def return_inf_imb_causality(
        self,
        cause_present,
        effect_present,
        effect_future,
        weights,
        conditioning_present=None,
        k=1,
        period_cause=None,
        period_effect=None,
        period_conditioning=None,
    ):
        """Return the imbalances (weight * cause_present, effect_present) -> effect_future.

           When conditioning_present is not None, the first space is extended with an additional weight,
           resulting in (weight1 * cause_present, weight2 * conditioning_present, effect_present) -> effect_future.

        Args:
            cause_present (np.ndarray(float)): N x D1 matrix, putative driver system data set at time 0
            effect_present (np.ndarray(float)): N x D2 matrix, putative driven system data set at time 0
            effect_future (np.ndarray(float)): N x D2 matrix, putative driven system data set at time tau
            weights (list(float), np.ndarray(float)): scaling parameters for the variables at time 0
                (1D array if conditioning_present is None, 2D array of shape (n_weights,2) otherwise,
                where the first column is referred to 'cause_present' and the second one to 'conditioning_present')
            conditioning_present (np.ndarray(float): N x D3 matrix, conditioning system data set at time 0
            k (int): order of nearest neighbour considered for the calculation of the imbalance
            period_cause (int,float,np.ndarray(float)): periods of variables in 'cause_present'
            period_effect (int,float,np.ndarray(float)): periods of variables in 'effect_present' and 'effect_future'
            period_conditioning (int,float,np.ndarray(float)): periods of variables in 'conditioning_present'

        Returns:
            imbalances (np.ndarray(float)): the information imbalances for the different weights
        """
        if self.period is not None:
            print(
                f"WARNING: the period argument {self.period} set in the MetricComparisons class will be "
                + "ignored.\nSet the periodicity of the features using instead the keywords "
                + "'period_cause' and 'period_effect'."
            )
        if (
            cause_present.shape[0] != effect_present.shape[0]
            or cause_present.shape[0] != effect_future.shape[0]
        ):
            raise ValueError(
                "Number of points must be the same in 'cause_present','effect_present' and 'effect_future'!"
            )
        if (
            conditioning_present is not None
            and conditioning_present.shape[0] != cause_present.shape[0]
        ):
            raise ValueError(
                "Number of points in 'conditioning_present' and 'cause_present' do not match!"
            )

        dim_cause = cause_present.shape[1]
        dim_effect = effect_present.shape[1]
        dim_conditioning = (
            None if conditioning_present is None else conditioning_present.shape[1]
        )

        _, ranks_effect_future = compute_nn_distances(
            effect_future, self.maxk, self.metric, period_effect
        )

        imbalances = Parallel(n_jobs=self.n_jobs)(
            delayed(self._return_inf_imb_causality_target_rank)(
                cause_present,
                effect_present,
                ranks_effect_future,
                conditioning_present,
                weight,
                k,
                _return_period_present(
                    period_cause,
                    period_effect,
                    period_conditioning,
                    dim_cause,
                    dim_effect,
                    dim_conditioning,
                    weight,
                ),
            )
            for weight in weights
        )

        return imbalances

    def _return_inf_imb_causality_target_rank(
        self,
        cause_present,
        effect_present,
        ranks_effect_future,
        conditioning_present=None,
        weight=1,
        k=1,
        period_present=None,
    ):
        """Return the imbalance (weight * cause_present, effect_present) -> effect_future.

           When 'conditioning_present' is not None, the imbalance that is computed is
           (weight[0] * cause_present, weight[1] * conditioning_present, effect_present) -> effect_future.


        Args:
            cause_present (np.ndarray(float)): N x D1 matrix, putative driver system data set at time 0
            effect_present (np.ndarray(float)): N x D2 matrix, putative driven system data set at time 0
            ranks_effect_future (np.ndarray(float)): N x maxk matrix, putative driven system ranks at time tau
            conditioning_present (np.ndarray(float): N x D3 matrix, conditioning system data set at time 0
            weight (float or np.ndarray(float)): scaling parameter space at time 0; scalar number if
                conditioning_present is None, np.ndarray of shape (2,) otherwise
            k (int): order of nearest neighbour considered for the calculation of the imbalance
            period_present (np.ndarray(float)): periods of all features in space
                (weight*cause_present, effect_present) if 'conditioning_present' is None, or in space
                (weight[0] * cause_present, weight[1] * conditioning_present, effect_present) otherwise

        Returns:
            imb (float): the information imbalance
        """
        if conditioning_present is None:
            space_present = np.column_stack((weight * cause_present, effect_present))
        else:
            space_present = np.column_stack(
                (
                    weight[0] * cause_present,
                    weight[1] * conditioning_present,
                    effect_present,
                )
            )

        _, ranks_present = compute_nn_distances(
            space_present,
            self.maxk,
            self.metric,
            period_present,
        )

        imb = _return_imbalance(ranks_present, ranks_effect_future, self.rng, k=k)

        return imb

    def return_inf_imb_causality_conditioning(
        self,
        cause_present,
        effect_present,
        conditioning_present,
        effect_future,
        weights_cause,
        weights_conditioning,
        k=1,
        period_cause=None,
        period_effect=None,
        period_conditioning=None,
    ):
        """Return the scanned imbalances in presence and in absence of the putative causal system.

        Args:
            cause_present (np.ndarray(float)): N x D1 matrix, putative driver system data set at time 0
            effect_present (np.ndarray(float)): N x D2 matrix, putative driven system data set at time 0
            conditioning_present (np.ndarray(float)): N x D3 matrix, conditioning driven system data set at time 0
            effect_future (np.ndarray(float)): N x D2 matrix, putative driven system data set at time tau
            weights_cause (list(float), np.ndarray(float)): scaling parameters for the causal variables
            weights_conditioning (list(float), np.ndarray(float)): scaling parameters for the conditioning variables
            k (int): order of nearest neighbour considered for the calculation of the imbalance
            period_cause (int,float,np.ndarray(float)): periods of variables in 'cause_present'
            period_effect (int,float,np.ndarray(float)): periods of variables in 'effect_present' and 'effect_future'
            period_conditioning (int,float,np.ndarray(float)): periods of variables in 'conditioning_present'

        Returns:
            imbs_no_cause (np.ndarray(float)): array of shape (weights_conditioning,) containing the imbalances
                (weight*cause_present, effect_present) -> effect_future
            imbs_with_cause (np.ndarray(float)): array of shape (weights_cause * weights_conditioning,) containing the
                imbalances (weight * cause_present, weight_conditioning * conditioning_present, effect_present)
                -> effect_future
        """
        weights_grid = _compute_2d_grid(weights_cause, weights_conditioning)

        d = MetricComparisons(maxk=cause_present.shape[0] - 1, n_jobs=self.n_jobs)

        imbs_no_cause = d.return_inf_imb_causality(
            cause_present=conditioning_present,
            effect_present=effect_present,
            effect_future=effect_future,
            weights=weights_conditioning,
            k=k,
            period_cause=period_conditioning,
            period_effect=period_effect,
        )
        imbs_with_cause = d.return_inf_imb_causality(
            cause_present=cause_present,
            effect_present=effect_present,
            conditioning_present=conditioning_present,
            effect_future=effect_future,
            weights=weights_grid,
            k=k,
            period_cause=period_cause,
            period_effect=period_effect,
            period_conditioning=period_conditioning,
        )

        return imbs_no_cause, imbs_with_cause

    def return_ranks_present_for_all_weights(
        self,
        cause_present,
        effect_present,
        weights,
        period_cause=None,
        period_effect=None,
    ):
        """Return the nearest neighbors' indices in space (weight*cause_present, effect_present) for all weights.

        Args:
            cause_present (np.ndarray(float)): N x D1 matrix, putative driver system data set at time 0
            effect_present (np.ndarray(float)): N x D2 matrix, putative driven system data set at time 0
            weights (list(float), np.ndarray(float)): scaling parameters for the driver system at time 0
            period_cause (int,float,np.ndarray(float)): periods of variables in 'cause_present'
            period_effect (int,float,np.ndarray(float)): periods of variables in 'effect_present'
        Returns:
            ranks_present (np.ndarray(float)): array of shape (N_weights, N, maxk+1), containing N_weights
                matrices (N, maxk+1) corresponding to the values of the scaling parameters in 'weights'
        """
        if self.period is not None:
            print(
                f"WARNING: the period argument {self.period} set in the MetricComparisons class will be "
                + "ignored.\nSet the periodicity of the features using instead the keyword "
                + "'period_present'."
            )
        if cause_present.shape[0] != effect_present.shape[0]:
            raise ValueError(
                "Number of points must be the same in 'cause_present','effect_present' and 'effect_future'!"
            )
        dim_cause = cause_present.shape[1]
        dim_effect = effect_present.shape[1]

        ranks_present = Parallel(n_jobs=self.n_jobs)(
            delayed(compute_nn_distances)(
                np.column_stack((weight * cause_present, effect_present)),
                self.maxk,
                self.metric,
                _return_period_mixed(
                    period_cause, period_effect, dim_cause, dim_effect, weight, 1
                ),
            )
            for weight in weights
        )

        ranks_present = np.delete(np.array(ranks_present), [0], axis=1)
        ranks_present = ranks_present.reshape(
            (len(weights), cause_present.shape[0], self.maxk + 1)
        )

        return ranks_present

    def return_inf_imb_causality_input_rank(
        self, ranks_present, effect_future, k=1, period_effect=None
    ):
        """Return the imbalances (weight * cause_present, effect_present) -> effect_future.

        Args:
            ranks_present (np.ndarray(float)): array of shape (N_weights, N, maxk+1), containing N_weights
                matrices (N, maxk+1) corresponding to the scanned values of the scaling parameter
            effect_future (np.ndarray(float)): N x D2 matrix, putative driven system data set at time tau
            k (int): order of nearest neighbour considered for the calculation of the imbalance
            period_effect (int,float,np.ndarray(float)): periods of the variables in 'effect_future'

        Returns:
            imbalances (np.ndarray(float)): the information imbalances for the different weights included
                in 'ranks_present'
        """
        if self.period is not None:
            print(
                f"WARNING: the period argument {self.period} set in the MetricComparisons class will be "
                + "ignored.\nSet the periodicity of the features using instead the keyword "
                + "'period_effect'."
            )

        _, ranks_effect_future = compute_nn_distances(
            effect_future, self.maxk, self.metric, period_effect
        )

        imbalances = Parallel(n_jobs=self.n_jobs)(
            delayed(_return_imbalance)(
                ranks_present[i_weight], ranks_effect_future, self.rng, k=k
            )
            for i_weight in range(ranks_present.shape[0])
        )

        return np.array(imbalances)
