import numpy as np
from sklearn.neighbors import NearestNeighbors

import multiprocessing
cores = multiprocessing.cpu_count()

class Base:

    def __init__(self, coordinates=None, distances=None, maxk=None, verbose=False, njobs=cores):

        self.X = coordinates
        self.maxk = maxk

        if coordinates is not None:
            self.Nele = coordinates.shape[0]
            self.distances = None
            if self.maxk is None:
                self.maxk = self.Nele - 1

        if distances is not None:
            if isinstance(distances, tuple):
                assert (distances[0].shape[0] == distances[1].shape[0])

                if self.maxk is None:
                    self.maxk = distances[0].shape[1] - 1

                self.Nele = distances[0].shape[0]
                self.distances = distances[0][:, :self.maxk + 1]
                self.dist_indices = distances[1][:, :self.maxk + 1]


            elif isinstance(distances, np.ndarray):
                assert (distances.shape[0] == distances.shape[1])  # assuming square matrix

                self.Nele = distances.shape[0]
                if self.maxk is None:
                    self.maxk = distances.shape[1] - 1

                self.dist_indices = np.asarray(np.argsort(distances, axis=1)[:, 0:self.maxk + 1])
                self.distances = np.asarray(
                    np.take_along_axis(distances, self.dist_indices, axis=1))

        self.verb = verbose
        self.njobs = njobs

    def compute_distances(self, maxk, njobs=1, metric='minkowski', p=2, algo='auto'):

        if self.maxk is None:
            self.maxk = maxk
        else:
            self.maxk = min(maxk, self.maxk)

        if self.verb: print('Computation of the distances up to {} NNs started'.format(self.maxk))

        nbrs = NearestNeighbors(n_neighbors=maxk, algorithm=algo, metric=metric, p=p,
                                n_jobs=njobs).fit(self.X)

        self.distances, self.dist_indices = nbrs.kneighbors(self.X)

        if self.verb: print('Computation of the distances finished')

    def remove_zero_dists(self):
        # TODO remove all the degenerate distances

        assert (self.distances is not None)

        # find all points with any zero distance
        indx = np.nonzero(self.distances[:, 1] < np.finfo(float).eps)

        # set nearest distance to eps:
        self.distances[indx, 1] = np.finfo(float).eps

        print(
            '{} couple of points where at 0 distance: their distance have been set to eps: {}'.format(
                len(indx), np.finfo(float).eps))
