from duly._base import Base

import numpy as np
from scipy.optimize import curve_fit
import duly.utils as ut

import multiprocessing
cores = multiprocessing.cpu_count()
rng = np.random.default_rng()

class IdEstimation(Base):

	"""Estimates the intrinsic dimension of a dataset choosing among various routines.

    Inherits from class Base

	Attributes:
		id_selected (int): (rounded) selected intrinsic dimension after each estimate. Parameter used for density estimation
		id_estimated_D (float):id estimated using Diego's routine
		id_estimated_2NN (float): 2NN id estimate

	"""

	def __init__(self, coordinates=None, distances=None, maxk=None, verbose=False, njobs=cores):
		super().__init__(coordinates=coordinates, distances=distances, maxk=maxk, verbose=verbose,
						 njobs=njobs)

	#---------------------------------------------------------------------------------------------- 

	def compute_id_diego(self, nneigh=1, fraction=0.95, d0=0.1, d1=1000):

		"""Compute intrinsic dimension through..?

		Args:
			nneigh: number of neighbours to be used for he estimate
			fraction: fraction of mus that will be considered for the estimate (discard highest mus)
			d0: lower estimate??
			d1: higher estimate??

		Returns:

		"""

		assert (self.distances is not None)
		if self.verb: print('ID estimation started, using nneigh = {}'.format(nneigh))

		# remove the 5% highest value of mu
		mu_full = self.distances[:, 2 * nneigh] / self.distances[:, 1 * nneigh]
		mu_full.sort()
		Nele_eff = int( len(mu_full) * fraction )
		mu = mu_full[:Nele_eff]

		f1 = ut._f(d1, mu, nneigh, Nele_eff)
		while (abs(d0 - d1) > 1.e-5):
			d2 = (d0 + d1) / 2.
			f2 = ut._f(d2, mu, nneigh, Nele_eff)
			if f2 * f1 > 0:
				d1 = d2
			else:
				d0 = d2

		d = (d0 + d1) / 2.

		self.id_estimated_D = d
		self.id_selected = np.rint(d)
		if self.verb: print('ID estimation finished, id selected = {}'.format(self.id_selected))

	#---------------------------------------------------------------------------------------------- 

	def compute_id_2NN(self, decimation=1, fraction=0.9, algorithm='base'):

		"""Compute intrinsic dimension using the 2NN algorithm

		Args:
			decimation: fraction of points used in the estimate. The lower the value, the less points used, the farther they are.\
						Can be used to change the scale at which one is looking at the data
			fraction: fraction of mus that will be considered for the estimate (discard highest mus)
			alghoritm: 'base' to perform the linear fit, 'ml' to perform maximum likelihood

		Returns:

		"""
		assert (0. < decimation and decimation <= 1.)
		# self.id_estimated_ml, self.id_estimated_ml_std = return_id(self.distances, decimation)
		# Nele = len(distances)
		# remove highest mu values
		dist_used = self.decimate(decimation,maxk=self.maxk)

		mus = np.log( dist_used[:, 2] / dist_used[:, 1] )
		
		Nele = dist_used.shape[0]
		Nele_eff = int(Nele * fraction)
		y = -np.log(1 - np.arange(1,Nele_eff+1)/Nele)
		mus_reduced = np.sort(mus)[:Nele_eff]
		
		#Nele_eff_dec = int(np.around(decimation * Nele_eff, decimals=0))
		#idxs = np.arange(Nele_eff)
		#idxs = np.random.choice(idxs, size=Nele_eff_dec, replace=False, p=None)
		#mus_reduced = mus[idxs]

		if algorithm == 'ml':
			id = Nele_eff / np.sum(mus_reduced)

		elif algorithm == 'base':

			def func(x, m):
				return m * x

			#y = np.array( [-np.log(1 - i / self.Nele) for i in range(1, Nele_eff + 1)] )
			#y = y[idxs]
			id, _ = curve_fit(func, mus_reduced, y)

		self.id_estimated_2NN = id[0]
		self.id_selected = np.around(id, decimals=2)[0]

		if self.verb:
			# print('ID estimated from ML is {:f} +- {:f}'.format(self.id_estimated_ml, self.id_estimated_ml_std))
			# print(f'Selecting ID of {self.id_selected}')
			print(f'ID estimation finished: selecting ID of {self.id_selected}')

	#---------------------------------------------------------------------------------------------- 

	def compute_id_gammaprior(self, alpha=2, beta=5):
		if self.distances is None: self.compute_distances()

		if self.verb: print(
			'ID estimation started, using alpha = {} and beta = {}'.format(alpha, alpha))

		distances_used = self.distances

		sum_log_mus = np.sum(np.log(distances_used[:, 2] / distances_used[:, 1]))

		alpha_post = alpha + self.Nele
		beta_post = beta + sum_log_mus

		mean_post = alpha_post / beta_post
		std_post = np.sqrt(alpha_post / beta_post ** 2)
		mode_post = (alpha_post - 1) / beta_post

		self.id_alpha_post = alpha_post
		self.id_beta_post = beta_post
		self.id_estimated_mp = mean_post
		self.id_estimated_mp_std = std_post
		self.id_estimated_map = mode_post
		self.id_selected = int(np.around(self.id_estimated_mp, decimals=0))

	#---------------------------------------------------------------------------------------------- 

	def set_id(self, id):
		self.id_selected = id

	#---------------------------------------------------------------------------------------------- 

if __name__ == '__main__':
	X = rng.uniform(size = (1000, 2))

	ide = IdEstimation(coordinates=X)

	ide.compute_distances(maxk = 10)

	ide.compute_id_2NN(decimation=1)

	print(ide.id_estimated_2NN,ide.id_selected)