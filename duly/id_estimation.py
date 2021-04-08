from duly._base import Base

import numpy as np
import math
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

	def __init__(self, coordinates=None, distances=None, maxk=30, verbose=False, njobs=cores, working_memory=1024):
		super().__init__(coordinates=coordinates, distances=distances, maxk=maxk, verbose=verbose,
						 njobs=njobs, working_memory=working_memory)

#----------------------------------------------------------------------------------------------
	#'better' way to perform scaling study of id
	def compute_id_scaling(self, range_max=1024, d0=0.001, d1=1000, return_ids = False, save_mus = False):

		range_r2 = min(range_max, self.Nele-1)
		max_step = int(math.log(range_r2, 2))
		steps = np.array([2**i for i in range(max_step)])

		distances, dist_indices, mus, r2s = self._get_mus_scaling(range_scaling=range_r2)

		#if distances have not been computed save them
		if self.distances is None:
			self.distances = distances
			self.dist_indices = dist_indices
			self.Nele = distances.shape[0]

		#array of ids (as a function of the average distange to a point)
		self.ids_scaling = np.empty(mus.shape[1])
		#array of error estimates (via fisher information)
		self.ids_scaling_std = np.empty(mus.shape[1])
		#array of average 'first' and 'second' neighbor distances, relative to each id estimate
		self.r2s_scaling = np.mean(r2s, axis = 0)

		#compute IDs via maximum likelihood (and their error) for all the scales up to range_scaling
		for i in range(mus.shape[1]):
			n1= 2**i
			id = ut._argmax_loglik(self.dtype, d0, d1, mus[:, i], n1, 2*n1, self.Nele, eps = 1.e-7)
			self.ids_scaling[i] = id
			self.ids_scaling_std[i] = (1/ut._fisher_info_scaling(id, mus[:, i], n1, 2*n1))**0.5

		#should we leave on option for saving mus? (can be useful especially for 'debugging' twoNN)
		if save_mus: self.mus_scaling = mus

		#shoud we leave an option to get the IdS?
		#The scale study should be done to do a quick plot and set the right id...
		if return_ids:
			return self.ids_scaling, self.ids_scaling_std, self.r2s_scaling

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

# if __name__ == '__main__':
#     X = rng.uniform(size = (1000, 2))
#
#     ide = IdEstimation(coordinates=X)
#
#     ide.compute_distances(maxk = 10)
#
#     ide.compute_id_2NN(decimation=1)
#
#     print(ide.id_estimated_2NN,ide.id_selected)
