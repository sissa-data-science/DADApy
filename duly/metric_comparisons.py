from joblib import Parallel, delayed

import duly.utils_.mlmax as mlmax
import duly.utils_.utils as ut

from duly._base import *


class MetricComparisons(Base):
    """This class contains several methods to compare metric spaces obtained using subsets of the data features.
    Using these methods one can assess whether two spaces are equivalent, completely independent, or whether one
    space is more informative than the other.

    Attributes:

    """

    def __init__(self, coordinates=None, distances=None, maxk=None, verbose=False, njobs=cores):
        super().__init__(coordinates=coordinates, distances=distances, maxk=maxk, verbose=verbose,
                         njobs=njobs)

    def return_inf_imb_of_coords(self, k=1, dtype='mean'):
        """Compute the information imbalances between all pairs of D features of the data

        Args:
            k (int): number of neighbours considered in the computation of the imbalances
            dtype (str): specific way to characterise the deviation from a delta distribution

        Returns:
            n_mat (np.array(float)): a DxD matrix containing all the information imbalances
        """
        assert (self.X is not None)

        ncoords = self.X.shape[1]

        n_mat = np.zeros((ncoords, ncoords))

        if self.njobs == 1:

            for i in range(ncoords):
                for j in range(i):
                    if self.verb == True: print('computing loss between coords ', i, j)

                    nij, nji = ut._return_imb_ij(i, j, self.maxk, self.X, k, dtype)
                    n_mat[i, j] = nij
                    n_mat[j, i] = nji

        elif self.njobs > 1:
            if self.verb is True: print(
                'computing imbalances with coord number on {} processors'.format(self.njobs))

            nmats = Parallel(n_jobs=self.njobs)(
                delayed(ut._return_imb_ij)(i, j, self.maxk, self.X, k, dtype) for i in range(ncoords) for j in range(i))

            indices = [(i, j) for i in range(ncoords) for j in range(i)]

            for idx, n in zip(indices, nmats):
                n_mat[idx[0], idx[1]] = n[0]
                n_mat[idx[1], idx[0]] = n[1]

        return n_mat

    def return_inf_imb_with_all_coords(self, k=1, dtype='mean'):
        """Compute the information imbalances between the original space and each one of its D features

        Args:
            k (int): number of neighbours considered in the computation of the imbalances
            dtype (str): specific way to characterise the deviation from a delta distribution

        Returns:
            (np.array(float)): a 2xD matrix containing the information imbalances between
            the original space and each of its D features.

        """
        assert (self.X is not None)

        ncoords = self.X.shape[1]

        if self.njobs == 1:
            n1s_n2s = []

            for i in range(ncoords):
                if self.verb is True: print('computing loss with coord number ', i)

                n0i, ni0 = ut._return_imb_with_coords(self.X, [i], self.dist_indices, self.maxk, k,
                                                      dtype)
                n1s_n2s.append((n0i, ni0))

        elif self.njobs > 1:
            if self.verb == True: print(
                'computing loss with coord number on {} processors'.format(self.njobs))
            n1s_n2s = Parallel(n_jobs=self.njobs)(
                delayed(ut._return_imb_with_coords)(self.X, [i], self.dist_indices, self.maxk, k,
                                                    dtype) for
                i in range(ncoords))
        else:
            raise ValueError("njobs cannot be negative")

        return np.array(n1s_n2s).T

    def return_inf_imb_with_selected_coords(self, coord_list, k=1, dtype='mean'):
        """Compute the information imbalances between the original space and a selection of features

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
        assert (self.X is not None)

        print('total number of computations is: ', len(coord_list))

        if self.njobs == 1:
            n1s_n2s = []

            for coords in coord_list:
                if self.verb == True: print('computing loss with coord selection')

                n0i, ni0 = ut._return_imb_with_coords(self.X, coords, self.dist_indices, self.maxk, k,
                                                      dtype)
                n1s_n2s.append((n0i, ni0))

        elif self.njobs > 1:
            if self.verb is True: print(
                'computing loss with coord number on {} processors'.format(self.njobs))
            n1s_n2s = Parallel(n_jobs=self.njobs)(
                delayed(ut._return_imb_with_coords)(self.X, coords, self.dist_indices, self.maxk, k,
                                                    dtype)
                for coords in coord_list)
        else:
            raise ValueError("njobs cannot be negative")

        return np.array(n1s_n2s).T

    def return_min_imbalance_between_selected_coords(self, coord_list1, coord_list2, k=1,
                                                     ltype='mean', params0=None):
        assert (self.X is not None)

        if params0 is None:

            # params0 = np.array(
            #     [1. / len(coord_list1)] * (len(coord_list1) - 1) + [1. / len(coord_list2)] * (
            #             len(coord_list2) - 1))

            params0 = np.random.uniform(0, 1, size=(len(coord_list1) + len(coord_list2)))
        else:

            assert (params0.shape[0] == len(coord_list1) + len(coord_list2))

        X1 = self.X[:, coord_list1]
        X2 = self.X[:, coord_list2]

        params, minimb = mlmax.Min_Symm_Imbalance(X1, X2, self.maxk, k, ltype, params0)

        return params, minimb

    def iterative_construction_of_sparse_representation(self, n_coords, k, ltype='mean'):
        print('taking dataset as the complete representation')
        assert self.X is not None

        D = self.X.shape[1]

        # select initial coordinate as the most informative
        losses = self.return_inf_imb_with_all_coords(k=k, dtype=ltype)
        proj = np.dot(losses.T, np.array([np.sqrt(.5), np.sqrt(.5)]))

        selected_coord = np.argmin(proj)

        print('1 coordinate selected: ', selected_coord)

        other_coords = list(np.arange(D).astype(int))

        other_coords.remove(selected_coord)

        selected_coords = [selected_coord]

        all_losses = [losses]

        np.savetxt('selected_coords.txt', selected_coords, fmt="%i")
        np.save('all_losses.npy', all_losses)

        for i in range(n_coords):
            coord_list = [selected_coords + [oc] for oc in other_coords]

            losses_ = self.return_inf_imb_with_selected_coords(coord_list, k=k, dtype=ltype)
            losses = np.empty((2, D))
            losses[:, :] = None
            losses[:, other_coords] = losses_

            proj = np.dot(losses_.T, np.array([np.sqrt(.5), np.sqrt(.5)]))

            selected_coord = other_coords[np.argmin(proj)]

            print('{} coordinate selected: '.format(i + 2), selected_coord)

            other_coords.remove(selected_coord)

            selected_coords.append(selected_coord)

            all_losses.append(losses)

            np.savetxt('selected_coords.txt', selected_coords, fmt='%i')
            np.save('all_losses.npy', all_losses)

        return selected_coords, all_losses

    def iterative_construction_of_sparse_representation_wlabel(self, n_coords, k, ltype='mean',
                                                               symm=True):
        print('taking labels as the reference representation')
        assert self.X is not None
        assert (self.gt_labels is not None)
        assert (type(self.gt_labels[0]) == np.ndarray)

        D = self.X.shape[1]

        # select ground truth label coordinate as the most informative
        losses = self.return_neigh_loss_of_labels_with_all_coords(k=k, ltype=ltype)

        if symm == True:
            proj = np.dot(losses.T, np.array([np.sqrt(.5), np.sqrt(.5)]))
            selected_coord = np.argmin(proj)
        else:
            selected_coord = np.argmin(losses[1])

        print('1 coordinate selected: ', selected_coord)

        other_coords = list(np.arange(D).astype(int))

        other_coords.remove(selected_coord)

        selected_coords = [selected_coord]

        all_losses = [losses]

        np.savetxt('selected_coords.txt', selected_coords, fmt="%i")
        np.save('all_losses.npy', all_losses)

        for i in range(n_coords):
            coord_list = [selected_coords + [oc] for oc in other_coords]

            losses_ = self.return_neigh_loss_of_labels_with_selected_coords(coord_list, k=k,
                                                                            ltype=ltype)
            losses = np.empty((2, D))
            losses[:, :] = None
            losses[:, other_coords] = losses_

            if symm == True:
                proj = np.dot(losses_.T, np.array([np.sqrt(.5), np.sqrt(.5)]))
                to_select = np.argmin(proj)
            else:
                to_select = np.argmin(losses_[1])

            selected_coord = other_coords[to_select]

            print('{} coordinate selected: '.format(i + 2), selected_coord)

            other_coords.remove(selected_coord)

            selected_coords.append(selected_coord)

            all_losses.append(losses)

            np.savetxt('selected_coords.txt', selected_coords, fmt="%i")
            np.save('all_losses.npy', all_losses)

        return selected_coords, all_losses

    def iterative_construction_of_sparse_representation_permsym(self, n_coords, k, nsym):
        print('taking dataset as the complete representation')
        assert self.X is not None

        D = self.X.shape[1]

        assert (D % nsym == 0)

        ncoords = int(D / nsym)
        print(D, nsym, ncoords)
        # select initial coordinate as the most informative
        coord_list = []

        for i in range(ncoords):
            sym_coord = [i + j * ncoords for j in range(nsym)]
            coord_list.append(sym_coord)

        losses = self.return_inf_imb_with_selected_coords(coord_list, k=k)

        proj = np.dot(losses.T, np.array([np.sqrt(.5), np.sqrt(.5)]))

        selected_coord = np.argmin(proj)

        print('1 coordinate selected: ', selected_coord)

        other_coords = list(np.arange(ncoords).astype(int))

        other_coords.remove(selected_coord)

        selected_coords = [selected_coord]

        all_losses = [losses]

        np.savetxt('selected_coords.txt', selected_coords, fmt="%i")

        for i in range(n_coords):

            coord_list_idx = [selected_coords + [oc] for oc in other_coords]

            coord_list = []

            for cli in coord_list_idx:
                sym_coords = []

                for i in cli:
                    sym_coord = [i + j * ncoords for j in range(nsym)]
                    sym_coords += sym_coord

                coord_list.append(sym_coords)

            losses = self.return_inf_imb_with_selected_coords(coord_list, k=k)

            proj = np.dot(losses.T, np.array([np.sqrt(.5), np.sqrt(.5)]))

            selected_coord = other_coords[np.argmin(proj)]

            print('{} coordinate selected: '.format(i + 2), selected_coord)

            other_coords.remove(selected_coord)

            selected_coords.append(selected_coord)

            all_losses.append(losses)

            np.savetxt('selected_coords.txt', selected_coords, fmt='%i')

            with open('all_losses.txt', 'w') as f:
                for i, loss in enumerate(all_losses):
                    f.write('### losses for coord ' + str(i + 1) + ' ###' + '\n')
                    for l in loss:
                        for ll in l:
                            f.write(str(ll))
                            f.write(' ')
                        f.write('\n')
                    f.write('\n')

        return selected_coords, all_losses

    def get_coordinates_infomation_ranking(self):
        print('taking dataset as the complete representation')
        assert self.X is not None

        d = self.X.shape[1]

        coords_kept = [i for i in range(d)]

        coords_removed = []

        projection_removed = []

        for i in range(d):
            print('niter is ', i)

            print(self.X.shape)
            nls = self.return_neigh_loss_with_coords(k=1)

            projection = nls.T.dot([np.sqrt(0.5), np.sqrt(0.5)])

            idx_to_remove = np.argmax(projection)

            c_to_remove = coords_kept[idx_to_remove]

            self.X = np.delete(self.X, idx_to_remove, 1)

            if i < (d - 1): self.compute_distances(maxk=self.Nele, njobs=1)

            coords_kept.pop(idx_to_remove)

            coords_removed.append(c_to_remove)

            projection_removed.append(projection[idx_to_remove])

            print('removing index number ', idx_to_remove)
            print('corresponding to coordinate number ', c_to_remove)

        # idx_to_remove = 0
        # c_to_remove = coords_kept[idx_to_remove]
        # coords_kept.pop(idx_to_remove)
        # projection_removed.append(projection[idx_to_remove])

        print('keeping datasets number ', coords_kept)

        return np.array(coords_removed)[::-1], np.array(projection_removed)[::-1]
