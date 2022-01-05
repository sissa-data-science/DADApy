import multiprocessing

import numpy as np
import matplotlib.pyplot as plt

# import scipy.special as sp
from scipy.optimize import minimize_scalar as SMin

from dadapy._base import Base
import dadapy.utils_.discrete_functions as df

# from dadapy.utils_ import utils as ut

cores = multiprocessing.cpu_count()
rng = np.random.default_rng()

class IdDiscrete(Base):
	"""Estimates the intrinsic dimension of a dataset with discrete features using a binomial likelihood.

	Inherits from class Base.

	Attributes:

		id_estimated_binom (float): id estimated using the binomial likelihood
		lk (int or int[:]): radii of the external shell
		ln (int or int[:]): radii of the internal shell
		k (int or or int[:]): total number of points within the external shell
		n (int or or int[:]): total number of points within the internal shell
		mask (bool[:]): eliminate points with bad statistics from the id estimation
		id_estimated_binom (double): intrinsic dimension obtained with the binomial estimator
		id_estimated_binom_std (double): error associated with the id estimation. computed through Cramer-Rao or Bayesian inference
		posterior_domain (double[:]): eventual support of the posterior distribution of the id
		posterior (double [:]): posterior distribution when evaluated with Bayesian inference

	"""

	def __init__(
		self,
		coordinates=None,
		distances=None,
		maxk=None,
		weights=None,
		verbose=False,
		njobs=cores,
	):
		super().__init__(
			coordinates=coordinates,
			distances=distances,
			maxk=maxk,
			verbose=verbose,
			njobs=njobs,
		)

		if weights is None:
			self.is_w = False
			self.weights = None
		else:
			self.set_w(weights)
			self.is_w = True

		self.lk = None
		self.ln = None
		self.k = None
		self.n = None
		self.mask = None

		self.id_estimated_binom = None
		self.id_estimated_binom_std = None
		self.posterior_domain = None
		self.posterior = None

	# ----------------------------------------------------------------------------------------------

	def fix_lk(self, lk=None, ln=None):
		"""Computes the k points within the given Rk and n points within given Rn.

		For each point, computes the number self.k of points within a sphere of radius Rk
		and the number self.n within an inner sphere of radius Rk. It also provides
		a mask to take into account those points for which the statistics might be wrong, i.e.
		k == self.maxk, meaning that all available points were selected or when we are not
		sure to have all the point of the selected shell. If self.maxk is equal
		to the number of points of the dataset no mask will be applied, as all the point
		were surely taken into account.
		Eventually add the weights of the points if they have an explicit multiplicity

		Args:
				lk (int): external shell radius
				ln (int): internal shell radius

		Returns:

		"""
		# checks-in and intialisations
		assert (
			self.distances is not None
		), "first compute distances with the proper metric (manhattan of hamming presumably)"

		if lk is not None and ln is not None:
			self.set_lk_ln(lk, ln)
		else:
			assert (
				self.lk is not None and self.ln is not None
			), "set self.lk and self.ln or insert proper values for the lk and ln parameters"

		# compute k and n
		if self.is_w is False:
			self.k = (self.distances <= self.lk).sum(axis=1)
			self.n = (self.distances <= self.ln).sum(axis=1)
		else:
			assert (
				self.weights is not None
			), "first insert the weights if you want to use them!"
			self.k = np.array(
				[
					sum(self.weights[self.dist_indices[i][el]])
					for i, el in enumerate(self.distances <= self.lk)
				],
				dtype=np.int,
			)
			self.n = np.array(
				[
					sum(self.weights[self.dist_indices[i][el]])
					for i, el in enumerate(self.distances <= self.ln)
				],
				dtype=np.int,
			)

		# checks-out
		if self.maxk == self.N-1:
			self.mask = np.ones(self.N, dtype=bool)
		else:
			# if not all possible NN were taken into account (i.e. maxk < N) and k is equal to self.maxk
			# or distances[:,-1]<lk, it is likely that there are other points within lk that are not being
			# considered and thus potentially altering the statistics -> neglect them through self.mask
			# in the calculation of likelihood
			self.mask = self.distances[:, -1] > self.lk  # or self.k == self.maxk

			if np.any(~self.mask):
				print(
					"NB: for "
					+ str(sum(~(self.mask)))
					+ " points, the counting of k could be wrong, "
					+ "as more points might be present within the selected Rk. In order not to affect "
					+ "the statistics a mask is provided to remove them from the calculation of the "
					+ "likelihood or posterior.\nConsider recomputing NN with higher maxk or lowering Rk."
				)

	# ----------------------------------------------------------------------------------------------

	def compute_id_binomial_lk(
		self, lk=None, ln=None, method="bayes", subset=None, plot=True
	):
		"""Calculate Id using the binomial estimator by fixing the eternal radius for all the points

		In the estimation of d one has to remove the central point from the counting of n and k
		as it generates the process but it is not effectively part of it

		Args:

				lk (int): radius of the external shell
				ln (int): radius of the internal shell
				subset (int): choose a random subset of the dataset to make the Id estimate
				method (str, default 'bayes'): choose method between 'bayes' and 'mle'. The bayesian estimate
					gives the distribution of the id, while mle only the max of the likelihood
				plot (bool): if bayes method is used, one can decide whether to plot the posterior
		"""
		# checks-in and initialisations
		assert (
			self.distances is not None
		), "first compute distances with the proper metric (manhattan of hamming presumably)"

		if lk is not None and ln is not None:
			self.set_lk_ln(lk, ln)
		else:
			assert (
				self.lk is not None and self.ln is not None
			), "set self.lk and self.ln or insert proper values for the lk and ln parameters"

		# routine
		self.fix_lk()

		# remove points with bad statistics (see fix_lk to see which are the conditions)
		n_eff = self.n[self.mask]
		k_eff = self.k[self.mask]
		if self.is_w:
			w_eff = self.weights[self.mask]


		if self.verb:
			print("n and k computed")

		# eventually perform the estimate using only a subset of the points (k and n are still computed on the whole dataset!!!)
		if subset is not None:
			# if subset is integer draw that amount of random numbers
			if isinstance(subset, (np.integer, int)):
				if subset < self.mask.sum():
					subset = rng.choice(len(n_eff), subset, replace=False, shuffle=False)
			# if subset is an array, use it as a mask
			elif isinstance(subset, np.ndarray):
				assert subset.shape[0] < self.mask.sum()
				assert isinstance(subset[0], (int, np.integer))

			n_eff = n_eff[subset]
			k_eff = k_eff[subset]
			if self.is_w:
				w_eff = w_eff[subset]

		# check statistics before performing id estimation
		if ~self.is_w:
			e_n = n_eff.mean()
			if e_n == 1.0:
				print(
					"no points in the inner shell, returning 0. Consider increasing lk and/or ln"
				)
				self.id_estimated_binom = 0
				return 0

		#choice of the method
		if method == "mle":
			# using weights?
			if self.is_w:
				N = w_eff.sum()
				k_eff = sum((k_eff-1)*w_eff)/float(N)
				n_eff = sum((n_eff-1)*w_eff)/float(N)
			else:
				w_eff = 1
				N = k_eff.shape[0]
				k_eff = k_eff.mean()-1
				n_eff = n_eff.mean()-1

			# explicit computation of likelihood, not necessary when ln and lk are fixed
			# self.id_estimated_binom = SMin(
			# 	df.compute_binomial_logL,
			# 	args=(self.lk, k_eff - 1, self.ln, n_eff - 1, True, w_eff),
			# 	bounds=(df.D_MIN+np.finfo(np.float16).eps, df.D_MAX),
			# 	method="bounded",
			# ).x

			self.id_estimated_binom = df.find_d(self.ln,self.lk,n_eff,k_eff)

			self.id_estimated_binom_std = df.Cramer_Rao(d=self.id_estimated_binom,ln=self.ln,lk=self.lk,N=N,k=k_eff)**0.5

			'''
			from scipy.optimize import root_scalar as rs
			n_eff = w_eff*(n_eff-1)
			k_eff = w_eff*(k_eff-1)

			self.id_estimated_binom = rs(df.find_d,args=(self.ln,self.lk,n_eff.mean(),k_eff.mean()),bracket=(df.D_MIN,df.D_MAX),method='brentq').root
			if is None:
				N = self.mask.sum()
			else
				N = subset
			self.id_estimated_binom_std = self.CramerRao(N=N,k=k_eff.mean())
			'''

		elif method == "bayes":
			if self.is_w:
				k_tot = self.weights[self.mask] * (k_eff - 1)
				n_tot = self.weights[self.mask] * (n_eff - 1)
			else:
				k_tot = k_eff - 1
				n_tot = n_eff - 1

			if self.verb:
				print("starting bayesian estimation")

			(
				self.id_estimated_binom,
				self.id_estimated_binom_std,
				self.posterior_domain,
				self.posterior,
			) = df._beta_prior_d(
				k_tot, n_tot, self.lk, self.ln, plot=plot, verbose=self.verb
			)
		else:
			print("select a proper method for id computation")
			return 0

	# ----------------------------------------------------------------------------------------------

	def fix_k(self, k_eff=None, ratio=0.5):
		"""Computes Rk, Rn, n for each point given a selected value of k

		This routine computes external radii lk, internal radii ln and internal points n.
		It also ensure that lk is scaled onto the most external shell
		for which we are sure to have all points.
		NB As a consequence the k will be point dependent

		Args:
				k_eff (int, default=self.k_max): selected (max) number of NN
				ratio (float, default = 0.5): approximate ratio among ln and lk

		Returns:
		"""

		# TODO: what if we have multiplicity???? use it straightly on the plain points
		#       or take it into account when counting the number of NN?

		# checks-in and initialisations
		assert (
			self.distances is not None
		), "first compute distances with the proper metric (manhattan of hamming presumably)"

		assert 0 < ratio < 1, "set a proper value for the ratio"

		if k_eff is None:
			k_eff = self.maxk-1
		else:
			assert (
				k_eff < self.maxk
			), "A k_eff > maxk was selected, recompute the distances with the proper amount on NN to see points up to that k_eff"

		# routine
		self.lk, self.k, self.ln, self.n = (
			np.zeros(self.N),
			np.zeros(self.N, dtype=np.int64),
			np.zeros(self.N),
			np.zeros(self.N, dtype=np.int64),
		)
		self.mask = np.ones(self.N, dtype=bool)

		# cut distances at the k-th NN and cycle over each point
		for i, dist_i in enumerate(self.distances[:, :k_eff+1]):

			# case 1: all possible neighbours have been considered, no extra possible unseen points
			if k_eff == self.N:
				lk_temp = dist_i[-1]
				k_temp = k_eff

			# case 2: only some NN are considered -> work on surely "complete" shells
			else:
				# Group distances according to shell. Returns radii of shell and cumulative number of points at that radius.
				# The index at which a distance is first met == number of points at smaller distance
				# EXAMPLE:
				# a = np.array([0,1,1,2,3,3,4,4,4,6,6])     and suppose it would go on as 6,6,6,7,8...
				# un, ind = np.unique(a, return_index=True)
				# un: array([0, 1, 2, 3, 4, 6])
				# ind: array([0, 1, 3, 4, 6, 9])
				unique, index = np.unique(dist_i, return_index=True)

				if unique.shape[0] < 3:
					# lk=0 may happen when we have smtg like dist_i = [0,0,0,0,0] or dist_i = [0,3,3,3,3,3]. As we don't have the full
					# information for at least two shells, we skip the point
					self.mask[i] = False
					continue

				# set lk to the distance of the second-to-last shell, for which we are sure to have full info
				lk_temp = unique[-2]  # EX: 4
				# set k to cumulative up to last complete shell
				k_temp = index[-1]  # EX: 9

			# fix internal radius
			ln_temp = np.rint(lk_temp * ratio)
			# if the inner shell is accidentally the same of the outer, go to the innerer one
			if ln_temp == lk_temp:
				ln_temp -= 1

			n_temp = sum(dist_i <= ln_temp)

			self.lk[i] = lk_temp
			self.k[i] = k_temp.astype(np.int64)
			self.n[i] = n_temp.astype(np.int64)
			self.ln[i] = ln_temp

		# checks out
		if any(~self.mask):
			print(
				"BE CAREFUL: "
				+ str(sum(~self.mask))
				+ " points would have Rk set to 0 "
				+ "and thus will be kept out of the statistics. Consider increasing k."
			)

	# --------------------------------------------------------------------------------------

	def fix_k_shell(self, k_shell, ratio):

		"""Computes the lk, ln, n given k_shell

		This routine computes the external radius lk, the associated points k, internal radius ln and associated points n.
		The computation is performed starting from a given number of shells.
		It ensure that Rk is scaled onto the most external shell for which we are sure to have all points.
		NB in this case we will have an effective, point dependent k

		Args:
				k_shell (int): selected (max) number of considered shells
				ratio (float): ratio among Rn and Rk

		"""

		# TODO: one might want to use it even with less shells available

		# initial checks
		assert (
			self.distances is not None
		), "first compute distances with the proper metric (manhattan of hamming presumably)"

		assert 0 < ratio < 1, "set a proper value for the ratio"

		assert k_shell < self.maxk, "asking for a much too large number of shells"

		self.lk, self.k, self.ln, self.n = (
			np.zeros(self.N),
			np.zeros(self.N, dtype=np.int64),
			np.zeros(self.N),
			np.zeros(self.N, dtype=np.int64),
		)
		self.mask = np.ones(self.N, dtype=bool)

		for i, dist_i in enumerate(self.distances):

			unique, index = np.unique(dist_i, return_index=True)

			# check whether the point has enough shells, at least one more than the one we want to consider
			if unique.shape[0] < k_shell + 1:
				self.mask[i] = False
				continue  # or lk_temp = unique[-1] even if the shell is not the wanted one???

			# set lk to the distance of the selected shell
			lk_temp = unique[k_shell]
			# and ln according to the ratio
			ln_temp = np.rint(lk_temp * ratio)
			# if the inner shell is accidentally the same of the outer, go to the innerer one
			if ln_temp == lk_temp:
				ln_temp -= 1

			# compute k and n
			if self.is_w:
				which_k = dist_i <= lk_temp
				self.k[i] = sum(self.weights[self.dist_indices[i][which_k]]).astype(
					np.int64
				)
				which_n = dist_i <= ln_temp
				self.n[i] = sum(self.weights[self.dist_indices[i][which_n]]).astype(
					np.int64
				)
			else:
				self.k[i] = index[k_shell + 1].astype(np.int64)
				self.n[i] = sum(dist_i <= ln_temp).astype(np.int64)

			self.lk[i] = lk_temp
			self.ln[i] = ln_temp

		# checks out
		if any(~self.mask):
			print(
				"BE CAREFUL: "
				+ str(sum(~self.mask))
				+ " points would have Rk set to 0 "
				+ "and thus will be kept out of the statistics. Consider increasing k."
			)

	# --------------------------------------------------------------------------------------

	def compute_id_binomial_k(self, k, shell=False, ratio=None, subset=None):
		"""Calculate Id using the binomial estimator by fixing the number of neighbours or shells

		As in the case in which one fix lk, also in this version of the estimation
		one removes the central point from n and k. Two different ways of computing
		k,n,lk,ln are available, whether one intends k as the k-th neighbour
		or the k-th shell. In both cases, one wants to be sure that the shell
		chosen is

		Args:
				k (int): order of neighbour that set the external shell
				shell (bool): k stands for number of neighbours or number of occupied shells
				ratio (float): ratio between internal and external shell
				subset (int): choose a random subset of the dataset to make the Id estimate

		"""
		# checks-in and initialisations
		assert (
			self.distances is not None
		), "first compute distances with the proper metric (manhattan of hamming presumably)"

		assert 0 < ratio < 1, "set a proper value for the ratio"

		if shell:
			self.fix_k_shell(k, ratio)

		else:
			self.fix_k(k, ratio)

		n_eff = self.n[self.mask]
		k_eff = self.k[self.mask]
		ln_eff = self.ln[self.mask]
		lk_eff = self.lk[self.mask]

		if subset is not None:
			assert isinstance(
				subset, (np.integer, int)
			), "subset needs to be an integer"
			if subset < len(n_eff):
				subset = rng.choice(len(n_eff), subset, replace=False, shuffle=False)
				n_eff = n_eff[subset]
				k_eff = k_eff[subset]
				ln_eff = ln_eff[subset]
				lk_eff = lk_eff[subset]

		e_n = n_eff.mean()
		if e_n == 1.0:
			print(
				"no points in the inner shell, returning 0. Consider increasing lk and/or the ratio"
			)
			return 0

		if self.is_w:
			ww = self.weights[self.mask]
		else:
			ww = 1

		self.id_estimated_binom = SMin(
			df.compute_binomial_logL,
			args=( lk_eff, k_eff - 1, ln_eff, n_eff - 1, True, ww,),
			bounds=(df.D_MIN+np.finfo(np.float16).eps, df.D_MAX),
			method="bounded",
		).x

	# ----------------------------------------------------------------------------------------------

	def model_validation(self, alpha=0.05, plot_pdf=False,plot_cdf=True):
		"""Use Kolmogorov-Smirnoff test to asses the goodness of the estimate

		In order to validate estimate of the intrinsic dimension and the model
		and the goodness of the binomial estimator we perform a KS test. In
		particular, once the ID has been computed, we generate a new set n_i
		starting from the k_i, lk_, ln_i and d using the binomial distribution
		(ln and lk can be scalars if the id estimate has been performed with
		fixed radii).
		We then compare the CDF obtained from both the n_empirical and the new
		n_i using the KS test, looking at the maximum distance between the two
		CDF

		Args:
				plot_pdf (bool, default=False): plot histogram of n_emp and n_i
				plot_cdf (bool, default=False): plot cdf of n_emp and n_i
		"""

		if self.id_estimated_binom is None:
			print('compute the id before validating the model!')
			return 0

		if isinstance(self.ln,np.ndarray):
			p = df.compute_discrete_volume(self.ln[self.mask],self.id_estimated_binom)/df.compute_discrete_volume(self.lk[self.mask],self.id_estimated_binom)
			n_model = rng.binomial(self.k[self.mask]-1,p)
		else:
			p = df.compute_discrete_volume(self.ln,self.id_estimated_binom)/df.compute_discrete_volume(self.lk,self.id_estimated_binom)
			n_model = rng.binomial(self.k[self.mask]-1,p)

		n_emp = self.n[self.mask] - 1

		from scipy.stats import ks_2samp as KS
		from scipy.stats import chisquare as X2

		s,pv = KS(n_emp,n_model)

		print("ks_stat:\t" + str(s))
		print("p-value:\t" + str(pv))

		if pv > alpha:
			print("We cannot reject the null hypotesis: the empirical and theoretical distributions has to be considered equivalent")
		else:
			print("We have to reject the null hypotesis: the two distributions ae not equivalent and thus the model as it is cannot be used to infer the id")

		# chisquare statistic, not working properly
		# a = np.histogram(n_emp,range=(-0.5,self.k.max()+0.5),bins=self.k.max()+1)
		# b = np.histogram(n_model,range=(-0.5,self.k.max()+0.5),bins=self.k.max()+1)
		# msk = np.where(b[0]>0)[0]
		# print(msk)
		# sX, pvX = X2(a[0][msk],b[0][msk])
		#
		# print("X_stat:\t" + str(sX))
		# print("X2p-value:\t" + str(pvX))
		#
		# if pvX > alpha:
		# 	print("We cannot reject the null hypotesis: the empirical and theoretical distributions has to be considered equivalent")
		# else:
		# 	print("We have to reject the null hypotesis: the two distributions ae not equivalent and thus the model as it is cannot be used to infer the id")


		if plot_pdf:

			a = np.histogram(n_emp,range=(-0.5,self.k.max()+0.5),bins=self.k.max()+1)
			b = np.histogram(n_model,range=(-0.5,self.k.max()+0.5),bins=self.k.max()+1)

			x = a[1][:-1]+0.5
			plt.figure()
			plt.plot(x,a[0],label='n_empirical')
			plt.plot(x,b[0],label='n_model')
			plt.legend()
			plt.show()

		if plot_cdf:

			a = np.histogram(n_emp,range=(-0.5,self.k.max()+0.5),bins=self.k.max()+1)
			b = np.histogram(n_model,range=(-0.5,self.k.max()+0.5),bins=self.k.max()+1)
			cdf_emp = np.cumsum(a[0])/self.mask.sum()
			cdf_mod = np.cumsum(b[0])/self.mask.sum()

			x = a[1][:-1]+0.5
			plt.figure()
			plt.plot(x,cdf_emp,label='cdf_empirical')
			plt.plot(x,cdf_mod,label='cdf_model')
			plt.legend()
			plt.show()

			diff = max(abs(cdf_emp-cdf_mod))
			print(diff,diff*np.sqrt(self.mask.sum()*0.5))

	# ----------------------------------------------------------------------------------------------

	def set_id(self, d):
		assert d > 0, "cannot support negative dimensions (yet)"
		self.id_selected = d

	# ----------------------------------------------------------------------------------------------

	def set_lk_ln(self, lk, ln):
		assert isinstance(ln, (np.int, np.int8, np.int16, np.int32, np.int64, int)) and ln > 0, "select a proper integer ln>0"
		assert isinstance(lk, (np.int, np.int8, np.int16, np.int32, np.int64, int)) and lk > 0, "select a proper integer lk>0"
		assert lk > ln, "select lk and ln, s.t. lk > ln"
		self.ln = ln
		self.lk = lk

	# ----------------------------------------------------------------------------------------------

	def set_w(self, w):
		assert len(w) == self.N and all(
			[wi > 0 and isinstance(wi, (np.int, int))] for wi in w
		), "load proper integer weights"
		self.weights = np.array(w, dtype=np.int)


# ----------------------------------------------------------------------------------------------


# if __name__ == '__main__':
#     X = rng.uniform(size = (1000, 2))
#
#     ide = IdEstimation(coordinates=X)
#
#     ide.compute_distances(maxk = 10)
#
#     ide.compute_id_2NN(decimation=1)
#
#     print(ide.id_estimated_2NN,ide.intrinsic_dim)
