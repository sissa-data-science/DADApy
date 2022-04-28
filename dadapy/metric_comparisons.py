# Copyright 2021-2022 The DADApy Authors. All Rights Reserved.
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

import numpy as np
from joblib import Parallel, delayed

from dadapy._utils.metric_comparisons import _return_imbalance
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
        njobs=cores,
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
            njobs (int): number of cores to be used
        """
        super().__init__(
            coordinates=coordinates,
            distances=distances,
            maxk=maxk,
            period=period,
            verbose=verbose,
            njobs=njobs,
        )

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

        imb_ij = _return_imbalance(dist_indices_i, dist_indices_j, k=k)
        imb_ji = _return_imbalance(dist_indices_j, dist_indices_i, k=k)

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
                    self.njobs
                )
            )

        nmats = Parallel(n_jobs=self.njobs)(
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
                "computing loss with coord number on {} processors".format(self.njobs)
            )

        n1s_n2s = Parallel(n_jobs=self.njobs)(
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
            k (int): order of nearest neighbour considered, default is 1

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
            X_, self.maxk, self.metric, period_
        )

        imb_coords_full = _return_imbalance(dist_indices_coords, dist_indices, k=k)
        imb_full_coords = _return_imbalance(dist_indices, dist_indices_coords, k=k)

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

    def return_label_overlap(self, labels, k=30):
        """Return the neighbour overlap between the full space and a set of labels.

        An overlap of 1 means that all neighbours of a point have the same label as the central point.

        Args:
            labels (np.ndarray): the labels with respect to which the overlap is computed
            k (int): the number of neighbours considered for the overlap

        Returns:
            (float): the neighbour overlap of the points
        """
        if self.distances is None:
            assert self.X is not None
            self.compute_distances()

        overlaps = []
        for i in range(self.N):
            neigh_idx_i = self.dist_indices[i, 1 : k + 1]
            overlaps.append(sum(labels[neigh_idx_i] == labels[i]) / k)

        overlap = np.mean(overlaps)
        return overlap

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
        assert self.X is not None

        X_ = self.X[:, coords]

        _, dist_indices_ = compute_nn_distances(X_, self.maxk, self.metric, self.period)

        overlaps = []
        for i in range(self.N):
            neigh_idx_i = dist_indices_[i, 1 : k + 1]
            overlaps.append(sum(labels[neigh_idx_i] == labels[i]) / k)

        overlap = np.mean(overlaps)

        return overlap

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
        assert self.X is not None

        X1_ = self.X[:, coords1]
        X2_ = self.X[:, coords2]

        _, dist_indices1_ = compute_nn_distances(X1_, k + 2, self.metric, self.period)
        _, dist_indices2_ = compute_nn_distances(X2_, k + 2, self.metric, self.period)

        overlap = np.mean(dist_indices1_[:, 1 : k + 1] == dist_indices2_[:, 1 : k + 1])

        return overlap

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
        assert self.X is not None

        overlaps = []
        for coords in coord_list:
            if self.verb:
                print("computing overlap for coord selection")

            overlap = self.return_label_overlap_coords(labels, coords, k)
            overlaps.append(overlap)

        return overlaps
