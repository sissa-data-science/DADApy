import numpy as np
from sklearn.neighbors import NearestNeighbors

import multiprocessing
cores = multiprocessing.cpu_count()
rng = np.random.default_rng()

class Base:

	"""Base class containig data and distances

	Attributes:
		X (float[:,:]): the data points loaded into the object
		maxk (int): maximum number of neighbours to be calculated
		distances (float[:,:]): distances between points saved in X
		dist_indeces (int[:,:]): ordering of neighbours according to the ditances
		verb (bool): whether you want the code to speak or shut up
		njobs (int): number of cores to be used
		p (int): metric used to compute distances

	"""

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

	#----------------------------------------------------------------------------------------------

	def compute_distances(self, maxk, njobs=1, metric='minkowski', p=2, algo='auto'):
		"""Compute distances between points up to the maxk nearest neighbour

		Args:
			maxk: maximum number of neighbours for which distance is computed and stored
			njobs: number of processes
			metric: type of metric
			p: type of metric
			algo: type of algorithm used

		"""
		if self.maxk is None:
			self.maxk = maxk
		else:
			self.maxk = min(maxk, self.maxk)

		self.p = p

		if self.verb: print(f'Computation of the distances up to {self.maxk} NNs started')

		nbrs = NearestNeighbors(n_neighbors=self.maxk, algorithm=algo, metric=metric, p=p,
								n_jobs=njobs).fit(self.X)

		self.distances, self.dist_indices = nbrs.kneighbors(self.X)

		if self.verb: print('Computation of the distances finished')

	#----------------------------------------------------------------------------------------------

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

	#----------------------------------------------------------------------------------------------

	def decimate(self, decimation, maxk = None):
		"""Compute distances for a random subset of points

		Args:
			decimation (float): fraction of points to use

		Returns:
			distences of decimated dataset

		"""

		assert( 0. < decimation and decimation <= 1. )

		if decimation == 1.:
			assert(self.distances is not None)
			return self.distances
		else:
			if maxk is None:
				maxk = self.maxk

			Nele_dec = int( self.Nele * decimation )
			idxs = rng.choice(self.Nele, Nele_dec, replace=False)
			X_temp = self.X[idxs]
			nbrs = NearestNeighbors(n_neighbors=maxk, p=self.p,
								n_jobs=self.njobs).fit(X_temp)
			self.dist_dec, self.ind_dec = nbrs.kneighbors(X_temp)
			return self.dist_dec

	#----------------------------------------------------------------------------------------------


if __name__ == '__main__':
	X = rng.uniform(size = (100, 2))

	base = Base(coordinates=X)

	base.compute_distances(maxk = 10)
	print(base.distances.shape)
	print(base.distances[0])

	dist_dec = base.decimate(0.5,10)

	print(dist_dec[0])
	print(dist_dec.shape)
