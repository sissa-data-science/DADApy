import multiprocessing

import numpy as np
from joblib import Parallel, delayed

import duly.utils_.utils as ut
from duly._base import Base
from duly.utils_.utils import compute_nn_distances

cores = multiprocessing.cpu_count()


class MetricComparisons(Base):
    """This class contains several methods to compare metric spaces obtained using subsets of the data features.
    Using these methods one can assess whether two spaces are equivalent, completely independent, or whether one
    space is more informative than the other.

    Attributes:

    """

    def __init__(
        self, coordinates=None, distances=None, maxk=None, verbose=False, njobs=cores
    ):
        super().__init__(
            coordinates=coordinates,
            distances=distances,
            maxk=maxk,
            verbose=verbose,
            njobs=njobs,
        )

    def return_inf_imb_two_selected_coords(self, coords1, coords2, k=1, dtype="mean"):
        """Returns the imbalances between distances taken as the i and the j component of the coordinate matrix X.

        Args:
            coords1 (list(int)): components for the first distance
            coords2 (list(int)): components for the second distance
            k (int): order of nearest neighbour considered for the calculation of the imbalance, default is 1
            dtype (str): type of information imbalance computation, default is 'mean'

        Returns:
            (float, float): the information imbalance from distance i to distance j and vice versa
        """
        X_ = self.X[:, coords1]
        _, dist_indices_i = compute_nn_distances(
            X_, self.maxk, self.metric, self.p, self.period
        )

        X_ = self.X[:, coords2]
        _, dist_indices_j = compute_nn_distances(
            X_, self.maxk, self.metric, self.p, self.period
        )

        imb_ij = ut._return_imbalance(dist_indices_i, dist_indices_j, k=k, dtype=dtype)
        imb_ji = ut._return_imbalance(dist_indices_j, dist_indices_i, k=k, dtype=dtype)

        return imb_ij, imb_ji

    def return_inf_imb_matrix_of_coords(self, k=1, dtype="mean"):
        """Compute the information imbalances between all pairs of D features of the data.

        Args:
            k (int): number of neighbours considered in the computation of the imbalances
            dtype (str): specific way to characterise the deviation from a delta distribution

        Returns:
            n_mat (np.array(float)): a DxD matrix containing all the information imbalances
        """
        assert self.X is not None

        ncoords = self.dims

        n_mat = np.zeros((ncoords, ncoords))

        if self.njobs == 1:

            for i in range(ncoords):
                for j in range(i):
                    if self.verb:
                        print("computing loss between coords ", i, j)

                    nij, nji = self.return_inf_imb_two_selected_coords(
                        [i], [j], k, dtype
                    )
                    n_mat[i, j] = nij
                    n_mat[j, i] = nji

        elif self.njobs > 1:
            if self.verb:
                print(
                    "computing imbalances with coord number on {} processors".format(
                        self.njobs
                    )
                )

            nmats = Parallel(n_jobs=self.njobs)(
                delayed(self.return_inf_imb_two_selected_coords)([i], [j], k, dtype)
                for i in range(ncoords)
                for j in range(i)
            )

            indices = [(i, j) for i in range(ncoords) for j in range(i)]

            for idx, n in zip(indices, nmats):
                print(indices, nmats)
                n_mat[idx[0], idx[1]] = n[0]
                n_mat[idx[1], idx[0]] = n[1]

        return n_mat

    def return_inf_imb_full_all_coords(self, k=1, dtype="mean"):
        """Compute the information imbalances between the 'full' space and each one of its D features

        Args:
            k (int): number of neighbours considered in the computation of the imbalances
            dtype (str): specific way to characterise the deviation from a delta distribution

        Returns:
            (np.array(float)): a 2xD matrix containing the information imbalances between
            the original space and each of its D features.

        """
        assert self.X is not None

        ncoords = self.X.shape[1]

        coord_list = [[i] for i in range(ncoords)]
        imbalances = self.return_inf_imb_full_selected_coords(
            coord_list, k=k, dtype=dtype
        )

        return imbalances

    def return_inf_imb_full_selected_coords(self, coord_list, k=1, dtype="mean"):
        """Compute the information imbalances between the 'full' space and a selection of features.

        Args:
            coord_list (list(list(int))): a list of the type [[1, 2], [8, 3, 5], ...] where each
            sub-list defines a set of coordinates for which the information imbalance should be
            computed.
            k (int): number of neighbours considered in the computation of the imbalances
            dtype (str): specific way to characterise the deviation from a delta distribution

        Returns:
            (np.array(float)): a 2xL matrix containing the information imbalances between
            the original space and each one of the L subspaces defined in coord_list

        """
        assert self.X is not None

        print("total number of computations is: ", len(coord_list))

        imbalances = self.return_inf_imb_target_selected_coords(
            self.dist_indices, coord_list, k=k, dtype=dtype
        )

        return imbalances

    def return_inf_imb_target_all_coords(self, target_ranks, k=1, dtype="mean"):
        """Compute the information imbalances between the 'target' space and a all single feature spaces in X.

        Args:
            target_ranks (np.array(int)): an array containing the ranks in the target space
            k (int): number of neighbours considered in the computation of the imbalances
            dtype (str): specific way to characterise the deviation from a delta distribution

        Returns:
            (np.array(float)): a 2xL matrix containing the information imbalances between
            the target space and each one of the L subspaces defined in coord_list

        """
        assert self.X is not None

        ncoords = self.dims

        coord_list = [[i] for i in range(ncoords)]
        imbalances = self.return_inf_imb_target_selected_coords(
            target_ranks, coord_list, k=k, dtype=dtype
        )

        return imbalances

    def return_inf_imb_target_selected_coords(
        self, target_ranks, coord_list, k=1, dtype="mean"
    ):
        """Compute the information imbalances between the 'target' space and a selection of features.

        Args:
            target_ranks (np.array(int)): an array containing the ranks in the target space
            coord_list (list(list(int))): a list of the type [[1, 2], [8, 3, 5], ...] where each
            sub-list defines a set of coordinates for which the information imbalance should be
            computed.
            k (int): number of neighbours considered in the computation of the imbalances
            dtype (str): specific way to characterise the deviation from a delta distribution

        Returns:
            (np.array(float)): a 2xL matrix containing the information imbalances between
            the target space and each one of the L subspaces defined in coord_list

        """
        assert self.X is not None
        assert target_ranks.shape[0] == self.X.shape[0]

        print("total number of computations is: ", len(coord_list))

        if self.njobs == 1:
            n1s_n2s = []
            for coords in coord_list:
                if self.verb:
                    print("computing loss with coord selection")
                n0i, ni0 = self._return_imb_with_coords(
                    self.X, coords, target_ranks, k, dtype
                )
                n1s_n2s.append((n0i, ni0))

        elif self.njobs > 1:
            if self.verb:
                print(
                    "computing loss with coord number on {} processors".format(
                        self.njobs
                    )
                )
            n1s_n2s = Parallel(n_jobs=self.njobs)(
                delayed(self._return_imb_with_coords)(
                    self.X, coords, target_ranks, k, dtype
                )
                for coords in coord_list
            )
        else:
            raise ValueError("njobs cannot be negative")

        return np.array(n1s_n2s).T

    def greedy_feature_selection_full(self, n_coords, k=1, dtype="mean", symm=True):
        """Greedy selection of the set of coordinates which is most informative about full distance measure.

        Args:
            n_coords: number of coodinates after which the algorithm is stopped
            k (int): number of neighbours considered in the computation of the imbalances
            dtype (str): specific way to characterise the deviation from a delta distribution
            symm (bool): whether to use the symmetrised information imbalance

        Returns:
            selected_coords:
            all_imbalances:
        """
        print("taking full space as the complete representation")
        assert self.X is not None
        selected_coords, all_imbalances = self.greedy_feature_selection_target(
            self.dist_indices, n_coords, k, dtype=dtype, symm=symm
        )

        return selected_coords, all_imbalances

    def greedy_feature_selection_target(
        self, target_ranks, n_coords, k, dtype="mean", symm=True
    ):
        """Greedy selection of the set of coordinates which is most informative about a target distance.

        Args:
            target_ranks (np.array(int)): an array containing the ranks in the target space
            n_coords: number of coodinates after which the algorithm is stopped
            k (int): number of neighbours considered in the computation of the imbalances
            dtype (str): specific way to characterise the deviation from a delta distribution
            symm (bool): whether to use the symmetrised information imbalance

        Returns:
            selected_coords:
            all_imbalances:
        """
        print("taking labels as the reference representation")
        assert self.X is not None

        dims = self.dims

        imbalances = self.return_inf_imb_target_all_coords(
            target_ranks, k=k, dtype=dtype
        )

        if symm:
            proj = np.dot(imbalances.T, np.array([np.sqrt(0.5), np.sqrt(0.5)]))
            selected_coord = np.argmin(proj)
        else:
            selected_coord = np.argmin(imbalances[1])

        print("1 coordinate selected: ", selected_coord)

        other_coords = list(np.arange(dims).astype(int))

        other_coords.remove(selected_coord)

        selected_coords = [selected_coord]

        all_imbalances = [imbalances]

        np.savetxt("selected_coords.txt", selected_coords, fmt="%i")
        np.save("all_losses.npy", all_imbalances)

        for i in range(n_coords):
            coord_list = [selected_coords + [oc] for oc in other_coords]

            imbalances_ = self.return_inf_imb_target_selected_coords(
                target_ranks, coord_list, k=k, dtype=dtype
            )
            imbalances = np.empty((2, dims))
            imbalances[:, :] = None
            imbalances[:, other_coords] = imbalances_

            if symm:
                proj = np.dot(imbalances_.T, np.array([np.sqrt(0.5), np.sqrt(0.5)]))
                to_select = np.argmin(proj)
            else:
                to_select = np.argmin(imbalances_[1])

            selected_coord = other_coords[to_select]

            print("{} coordinate selected: ".format(i + 2), selected_coord)

            other_coords.remove(selected_coord)

            selected_coords.append(selected_coord)

            all_imbalances.append(imbalances)

            np.savetxt("selected_coords.txt", selected_coords, fmt="%i")
            np.save("all_losses.npy", all_imbalances)

        return selected_coords, all_imbalances

    def return_inf_imb_target_all_dplets(self, target_ranks, d, k=1, dtype="mean"):
        """Compute the information imbalances between a target distance and all possible combinations of d coordinates
        contained of X.

        Args:
            target_ranks (np.array(int)): an array containing the ranks in the target space
            d:
            k (int): number of neighbours considered in the computation of the imbalances
            dtype (str): specific way to characterise the deviation from a delta distribution

        Returns:

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
            target_ranks, coord_list, k=k, dtype=dtype
        )

        return np.array(coord_list), np.array(imbalances)

    def return_coordinates_infomation_ranking(self):
        print("taking dataset as the complete representation")
        assert self.X is not None

        d = self.X.shape[1]

        coords_kept = [i for i in range(d)]

        coords_removed = []

        projection_removed = []

        for i in range(d):
            print("niter is ", i)

            print(self.X.shape)
            nls = self.return_inf_imb_full_all_coords(k=1)

            projection = nls.T.dot([np.sqrt(0.5), np.sqrt(0.5)])

            idx_to_remove = np.argmax(projection)

            c_to_remove = coords_kept[idx_to_remove]

            self.X = np.delete(self.X, idx_to_remove, 1)

            if i < (d - 1):
                self.compute_distances(maxk=self.N)

            coords_kept.pop(idx_to_remove)

            coords_removed.append(c_to_remove)

            projection_removed.append(projection[idx_to_remove])

            print("removing index number ", idx_to_remove)
            print("corresponding to coordinate number ", c_to_remove)

        # idx_to_remove = 0
        # c_to_remove = coords_kept[idx_to_remove]
        # coords_kept.pop(idx_to_remove)
        # projection_removed.append(projection[idx_to_remove])

        print("keeping datasets number ", coords_kept)

        return np.array(coords_removed)[::-1], np.array(projection_removed)[::-1]

    def return_label_overlap(self, labels, k=30):
        assert self.distances is not None

        overlaps = []
        for i in range(self.N):
            neigh_idx_i = self.dist_indices[i, 1 : k + 1]
            overlaps.append(sum(labels[neigh_idx_i] == labels[i]) / k)

        overlap = np.mean(overlaps)
        return overlap

    def return_label_overlap_coords(self, labels, coords, k=30):
        assert self.X is not None

        X_ = self.X[:, coords]

        _, dist_indices_ = compute_nn_distances(
            X_, self.maxk, self.metric, self.p, self.period
        )

        overlaps = []
        for i in range(self.N):
            neigh_idx_i = dist_indices_[i, 1 : k + 1]
            overlaps.append(sum(labels[neigh_idx_i] == labels[i]) / k)

        overlap = np.mean(overlaps)
        return overlap

    def return_label_overlap_selected_coords(self, labels, coord_list, k=30):
        assert self.X is not None

        if True:  # self.njobs == 1:
            overlaps = []
            for coords in coord_list:
                if self.verb:
                    print("computing overlap for coord selection")

                overlap = self.return_label_overlap_coords(labels, coords, k)
                overlaps.append(overlap)

        return overlaps

    def _return_imb_with_coords(self, X, coords, dist_indices, k, dtype="mean"):
        """Returns the imbalances between a 'full' distance computed using all coordinates, and an alternative distance
         built using a subset of coordinates.

        Args:
            X: coordinate matrix
            coords: subset of coordinates to be used when building the alternative distance
            dist_indices (int[:,:]): nearest neighbours according to full distance
            k (int): order of nearest neighbour considered, default is 1
            dtype (str): type of information imbalance computation, default is 'mean'

        Returns:
            (float, float): the information imbalance from 'full' to 'alternative' and vice versa
        """
        X_ = X[:, coords]

        _, dist_indices_coords = compute_nn_distances(
            X_, self.maxk, self.metric, self.p, self.period
        )

        imb_coords_full = ut._return_imbalance(
            dist_indices_coords, dist_indices, k=k, dtype=dtype
        )
        imb_full_coords = ut._return_imbalance(
            dist_indices, dist_indices_coords, k=k, dtype=dtype
        )
        print("computing imbalances with coords ", coords)
        return imb_full_coords, imb_coords_full
