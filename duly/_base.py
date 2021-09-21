import math
import time
import multiprocessing
from functools import partial

import numpy as np

from sklearn.metrics import pairwise_distances_chunked
from sklearn.neighbors import NearestNeighbors

from duly.utils_.utils import from_all_distances_to_nndistances
from duly.utils_.utils import compute_nn_distances

cores = multiprocessing.cpu_count()
rng = np.random.default_rng()


class Base:
    """Base class. A simple container of coordinates and/or distances and of basic methods.

    Attributes:
        N (int): number of data points
        X (np.ndarray(float)): the data points loaded into the object, of shape (N , dimension of embedding space)
        dims (int, optional): embedding dimension of the datapoints
        maxk (int): maximum number of neighbours to be considered for the calculation of distances
        distances (float[:,:]): A matrix of dimension N x mask containing distances between points
        dist_indices (int[:,:]): A matrix of dimension N x mask containing the indices of the nearest neighbours
        verb (bool): whether you want the code to speak or shut up
        njobs (int): number of cores to be used
        p (int): metric used to compute distances
        period (np.array(float), optional): array of shape (dims,) containing periodicity for each coordinate. Default is None
    """

    def __init__(
        self,
        coordinates=None,
        distances=None,
        maxk=None,
        verbose=False,
        njobs=cores,
    ):

        self.X = coordinates
        self.maxk = maxk
        self.verb = verbose
        self.njobs = njobs
        self.dims = None
        self.N = None
        self.metric = "euclidean"
        self.p = 2
        self.period = None

        if coordinates is not None:
            assert isinstance(self.X, np.ndarray), "Coordinates must be in numpy ndarray format"

            self.dtype = self.X.dtype

            self.N = self.X.shape[0]

            # self.N = coordinates.shape[0]
            self.dims = coordinates.shape[1]
            self.distances = None
            # BUG to be solved: the next line
            if self.maxk is None:
                self.maxk = self.N - 1

        if distances is not None:
            if isinstance(distances, tuple):
                assert distances[0].shape[0] == distances[1].shape[0]
                assert isinstance(distances[0], np.ndarray)
                assert isinstance(distances[1], np.ndarray)
                is_ndarray = isinstance(distances, np.ndarray)

                if self.maxk is None:
                    self.maxk = distances[0].shape[1] - 1

                self.N = distances[0].shape[0]

                self.distances = distances[0][:, : self.maxk + 1]

                self.dist_indices = (
                    distances[1][:, : self.maxk + 1]
                    if is_ndarray
                    else distances[1].numpy().shape[0]
                )

            else:
                assert (
                    distances.shape[0] == distances.shape[1]
                )  # assuming square matrix
                assert isinstance(distances, np.ndarray)

                self.N = distances.shape[0]
                if self.maxk is None:
                    self.maxk = distances.shape[1] - 1

                self.dist_indices, self.distances = from_all_distances_to_nndistances(
                    distances, self.maxk
                )

            self.dtype = self.distances.dtype

    # ----------------------------------------------------------------------------------------------

    def compute_distances(self, maxk=None, metric="minkowski", p=2, period=None):
        """Compute distaces between points up to the maxk nearest neighbour

        Args:
                maxk: maximum number of neighbours for which distance is computed and stored
                metric: type of metric
                p: type of metric
                period (float or np.array): periodicity (only used for periodic distance computation). Default is None.

        """
        if self.verb:
            print("Computation of distances started")
            sec = time.time()

        self.metric = metric
        self.p = p
        if period is not None:
            if isinstance(period, np.ndarray) and period.shape == (self.dims,):
                self.period = period
            elif isinstance(period, (int, float)):
                self.period = np.full((self.dims), fill_value=period, dtype=float)
            else:
                raise ValueError(
                    "'period' must be either a float scalar or a numpy array of floats of shape ({},)".format(
                        self.dims
                    )
                )

        if maxk is not None:
            self.maxk = maxk
        else:
            assert (
                self.maxk is not None
            ), "set parameter maxk in the function or for the class"

        if self.verb and period is not None:
            print(
                "Computing periodic distances.",
                "The coordinates are assumed to be in the range [0, period]",
            )

        if self.verb:
            print(f"Computation of the distances up to {self.maxk} NNs started")

        self.distances, self.dist_indices = compute_nn_distances(
            self.X, self.maxk, self.metric, self.p, self.period
        )

        sec2 = time.time()
        if self.verb:
            print("{0:0.2f} seconds for computing distances".format(sec2 - sec))

    # ---------------------------------------------------------------------------

    # better to use this formulation which can be applied to _mus_scaling_reduce_func
    def _remove_zero_dists(self, distances):

        # TO IMPROVE/CHANGE
        # to_remove = distances[:, 2] < np.finfo(self.dtype).eps
        # distances = distances[~to_remove]
        # indices = indices[~to_remove]

        # TO TEST

        # find all points with any zero distance
        indx_ = np.nonzero(distances[:, 1] < np.finfo(self.dtype).eps)[0]
        # set nearest distance to eps:
        distances[indx_, 1] = np.finfo(self.dtype).eps

        return distances

    # ----------------------------------------------------------------------------------------------

    def _mus_scaling_reduce_func(self, dist, start, range_scaling=None):
        """Compute

        adapted from kneighbors function of sklearn
        https://github.com/scikit-learn/scikit-learn/blob/95119c13af77c76e150b753485c662b7c52a41a2/sklearn/neighbors/_base.py

        Description.

        Args:
                range_scaling

        Returns:


        """

        max_step = int(math.log(range_scaling, 2))
        steps = np.array([2 ** i for i in range(max_step)])

        sample_range = np.arange(dist.shape[0])[:, None]
        neigh_ind = np.argpartition(dist, range_scaling - 1, axis=1)
        neigh_ind = neigh_ind[:, :range_scaling]

        # argpartition doesn't guarantee sorted order, so we sort again
        neigh_ind = neigh_ind[sample_range, np.argsort(dist[sample_range, neigh_ind])]

        dist = np.sqrt(dist[sample_range, neigh_ind])

        dist = self._remove_zero_dists(dist)
        mus = dist[:, steps[1:]] / dist[:, steps[:-1]]
        rs = dist[:, np.array([steps[:-1], steps[1:]])]

        return (
            dist[:, : self.maxk + 1],
            neigh_ind[:, : self.maxk + 1],
            mus,
            np.mean(rs, axis=1),
        )

    def _return_mus_scaling(self, range_scaling):
        """Compute

        Description.

        Args:
                range_scaling

        Returns:


        """

        reduce_func = partial(
            self._mus_scaling_reduce_func, range_scaling=range_scaling
        )

        # N0 = self.X.shape[0]
        # self.X = np.unique(self.X, axis=0)

        # self.N = self.X.shape[0]
        # if self.N != N0:
        #     print(
        #         f"{N0-self.N}/{N0} overlapping datapoints: keeping {self.N} unique elements"
        #     )

        kwds = {"squared": True}
        chunked_results = list(
            pairwise_distances_chunked(
                self.X,
                self.X,
                reduce_func=reduce_func,
                metric=self.metric,
                n_jobs=self.njobs,
                working_memory=1024,
                **kwds,
            )
        )

        neigh_dist, neigh_ind, mus, rs = zip(*chunked_results)
        return (
            np.vstack(neigh_dist),
            np.vstack(neigh_ind),
            np.vstack(mus),
            np.vstack(rs),
        )

    def remove_identical_points_TO_BE_WRITTEN(self):

        N0 = self.X.shape[0]
        # removal of overlapping data points

        self.X = np.unique(self.X)  # This does not work properly because of sorting!!

        self.N = self.X.shape[0]

        if self.N != N0:
            print(
                f"{N0 - self.N}/{N0} overlapping datapoints: keeping {self.N} unique elements"
            )
            if self.distances is not None:
                self.distances, self.dist_indices = None, None
