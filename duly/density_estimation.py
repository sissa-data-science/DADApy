import time
import multiprocessing

import numpy as np
from scipy.special import gammaln
from scipy import sparse
from scipy import linalg as slin

from duly.id_estimation import IdEstimation
from duly.cython_ import cython_maximum_likelihood_opt as cml
from duly.cython_ import cython_grads as cgr
from duly.cython_ import cython_density as cd
from duly.utils_.utils import compute_cross_nn_distances
from duly.utils_.density_estimation import return_density_kstarNN, return_density_PAk

cores = multiprocessing.cpu_count()


class DensityEstimation(IdEstimation):
    """Computes the log-density and its error at each point and other properties.

    Inherits from class IdEstimation. Can estimate the optimal number k* of neighbors for each points. \
    Can compute the log-density and its error at each point choosing among various kNN-based methods. \
    Can return an estimate of the gradient of the log-density at each point and an estimate of the error on each component. \
    Can return an estimate of the linear deviation from constant density at each point and an estimate of the error on each component. \
    
    TODO:
    - compute dc when compute_kstar or when set_kstar and nowhere else
    - implement self.kstar and self.fixedk of a bool attribute self.iskfixed which is True when kstar is an array of identical integers


    Attributes:
        kstar (np.array(float)): array containing the chosen number k* in the neighbourhood of each of the N points
        nspar (int): total number of edges in the directed graph defined by kstar (sum over all points of kstar minus N)
        nind_list (np.ndarray(int), optional): size nspar x 2. Each row is a couple of indices of the connected graph stored in order of increasing point index and increasing neighbour length (E.g.: in the first row (0,j), j is the nearest neighbour of the first point. In the second row (0,l), l is the second-nearest neighbour of the first point. In the last row (N-1,m) m is the kstar-1-th neighbour of the last point.)
        nind_iptr (np.array(int), optional): size N+1. For each elemen i stores the 0-th index in nind_list at which the edges starting from point i start. The last entry is set to nind_list.shape[0].
        common_neighs (scipy.sparse.csr_matrix(float), optional): stored as a sparse symmetric matrix of size N x N. Entry (i,j) gives the common number of neighbours between points i and j. Such value is reliable only if j is in the neighbourhood of i or vice versa.
        neigh_vector_diffs (np.ndarray(float), optional): stores vector differences from each point to its k*-1 nearest neighbors. Accessed by the method return_vector_diffs(i,j) for each j in the neighbourhood of i
        neigh_dists (np.array(float), optional): stores distances from each point to its k*-1 nearest neighbors in the order defined by nind_list
        dc (np.array(float), optional): array containing the distance of the k*th neighbor from each of the N points
        log_den (np.array(float), optional): array containing the N log-densities
        log_den_err (np.array(float), optional): array containing the N errors on the log_den
        grads (np.ndarray(float), optional): for each line i contains the gradient components estimated from from point i
        grads_var (np.ndarray(float), optional): for each line i contains the estimated variance of the gradient components at point i
        check_grads_covmat (bool, optional): it is flagged "True" when grads_var contains the variance-covariance matrices of the gradients.
        Fij_array (list(np.array(float)), optional): stores for each couple in nind_list the estimates of deltaF_ij computed from point i as semisum of the gradients in i and minus the gradient in j.
        Fij_var_array (np.array(float), optional): stores for each couple in nind_list the estimates of the squared errors on the values in Fij_array.

    
    Bello sto stile di documentazione:
    
    Parameters
    ----------
    X : {float, np.ndarray, or theano symbolic variable}
        X coordinate. If you supply an array, x and y need to be the same shape,
        and the potential will be calculated at each (x,y pair)
    y : {float, np.ndarray, or theano symbolic variable}
        Y coordinate. If you supply an array, x and y need to be the same shape,
        and the potential will be calculated at each (x,y pair)


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

        self.kstar = None
        self.nspar = None
        self.nind_list = None
        self.nind_iptr = None
        self.common_neighs = None
        self.neigh_vector_diffs = None
        self.neigh_dists = None
        self.dc = None
        self.log_den = None
        self.log_den_err = None
        self.grads = None
        self.grads_var = None
        self.check_grads_covmat = False
        self.Fij_array = None
        self.Fij_var_array = None

    # ----------------------------------------------------------------------------------------------

    def set_kstar(self, k=0):
        """Set all elements of kstar to a fixed value k. Reset all other class attributes (all depending on kstar).

        Args:
            k: number of neighbours used to compute the density

        Returns:

        """
        if isinstance(k, np.ndarray):
            self.kstar = k
        else:
            self.kstar = np.full(self.N, k, dtype=int)

        self.nspar = None
        self.nind_list = None
        self.nind_iptr = None
        self.common_neighs = None
        self.neigh_vector_diffs = None
        self.neigh_dists = None
        self.dc = None
        self.log_den = None
        self.log_den_err = None
        self.grads = None
        self.grads_var = None
        self.check_grads_covmat = False
        self.Fij_array = None
        self.Fij_var_array = None

    # ----------------------------------------------------------------------------------------------

    def compute_density_kNN(self, k=5):
        """Compute the density of of each point using a simple kNN estimator

        Args:
            k: number of neighbours used to compute the density

        Returns:

        """
        assert self.intrinsic_dim is not None

        if self.verb:
            print("k-NN density estimation started (k={})".format(k))

        # kstar = np.full(self.N, k, dtype=int)
        self.set_kstar(k)
        dc = np.zeros(self.N, dtype=float)
        log_den = np.zeros(self.N, dtype=float)
        log_den_err = np.zeros(self.N, dtype=float)
        prefactor = np.exp(
            self.intrinsic_dim / 2.0 * np.log(np.pi)
            - gammaln((self.intrinsic_dim + 2) / 2)
        )
        log_den_min = 9.9e300

        for i in range(self.N):
            dc[i] = self.distances[i, self.kstar[i]]
            log_den[i] = np.log(self.kstar[i]) - (
                np.log(prefactor)
                + self.intrinsic_dim * np.log(self.distances[i, self.kstar[i]])
            )

            log_den_err[i] = 1.0 / np.sqrt(self.kstar[i])
            if log_den[i] < log_den_min:
                log_den_min = log_den[i]

        # Normalise density
        log_den -= np.log(self.N)

        self.log_den = log_den
        self.log_den_err = log_den_err
        self.dc = dc
        # self.kstar = kstar

        if self.verb:
            print("k-NN density estimation finished")

    # ----------------------------------------------------------------------------------------------

    def compute_kstar(self, Dthr=23.92812698):
        """Computes an optimal choice of k for each point.
        Args:
            Dthr: Likelihood ratio parameter used to compute optimal k, the value of Dthr=23.92 corresponds
            to a p-value of 1e-6.
        Returns:
        """

        if self.intrinsic_dim is None:
            self.compute_id_2NN()

        if self.verb:
            print("kstar estimation started, Dthr = {}".format(Dthr))

        sec = time.time()


        kstar = cd._compute_kstar(
            self.intrinsic_dim,
            self.N,
            self.maxk,
            Dthr,
            self.dist_indices,
            self.distances,
        )
        self.set_kstar(kstar)

        sec2 = time.time()
        if self.verb:
            print("{0:0.2f} seconds computing kstar".format(sec2 - sec))

    # ----------------------------------------------------------------------------------------------

    def compute_neigh_indices(self):
        """Computes the indices of all the couples [i,j] such that j is a neighbour of i up to the k*-th nearest (excluded).
        The couples of indices are stored in a numpy ndarray of rank 2 and secondary dimension = 2.
        The index of the corresponding AAAAAAAAAAAAA make indpointer which is a np.array of length N which indicates for each i the starting index of the corresponding [i,.] subarray.

        """

        if self.kstar is None:
            self.compute_kstar()

        if self.verb:
            print("Computation of the neighbour indices started")

        sec = time.time()

        # self.get_vector_diffs = sparse.csr_matrix((self.N, self.N),dtype=np.int_)
        self.nind_list, self.nind_iptr = cgr.return_neigh_ind(
            self.dist_indices, self.kstar
        )

        self.nspar = len(self.nind_list)

        sec2 = time.time()
        if self.verb:
            print("{0:0.2f} seconds computing neighbour indices".format(sec2 - sec))

    # ----------------------------------------------------------------------------------------------

    def compute_neigh_dists(self):
        """Computes the (directed) neighbour distances graph using kstar[i] neighbours for each point i.
        Distances are stored in np.array form according to the order of nind_list. 

        """

        if self.distances is None:
            self.compute_distances()

        if self.kstar is None:
            self.compute_kstar()

        if self.verb:
            print("Computation of the neighbour distances started")

        sec = time.time()

        self.neigh_dists = cgr.return_neigh_distances_array(
            self.distances, self.dist_indices, self.kstar
        )

        sec2 = time.time()
        if self.verb:
            print("{0:0.2f} seconds computing neighbour distances".format(sec2 - sec))

    # ----------------------------------------------------------------------------------------------    
 
    def return_sparse_distance_graph(self):
        """Returns the (directed) neighbour distances graph using kstar[i] neighbours for each point i in N x N sparse csr_matrix form.

        """
 
        if self.neigh_dists is None:
            self.compute_neigh_dists()

        dgraph = sparse.lil_matrix((self.N, self.N),dtype=np.float_)


        
        for ind_spar, indices in enumerate(self.nind_list):
                dgraph[indices[0],indices[1]] = self.neigh_dists[ind_spar]

        return dgraph.tocsr()

    # ----------------------------------------------------------------------------------------------    

    def compute_neigh_vector_diffs(
        self,
    ):
        """Compute the vector differences from each point to its k* nearest neighbors.
        The resulting vectors are stored in a numpy ndarray of rank 2 and secondary dimension = dims.
        The index of  scipy sparse csr_matrix format.

        AAA better implement periodicity to take both a scalar and a vector

        """
        # compute neighbour indices
        if self.nind_list is None:
            self.compute_neigh_indices()

        if self.verb:
            print("Computation of the vector differences started")
        sec = time.time()

        # self.get_vector_diffs = sparse.csr_matrix((self.N, self.N),dtype=np.int_)
        if self.period is None:
            self.neigh_vector_diffs = cgr.return_neigh_vector_diffs(
                self.X, self.nind_list
            )
        else:
            self.neigh_vector_diffs = cgr.return_neigh_vector_diffs_periodic(
                self.X, self.nind_list, self.period
            )

        sec2 = time.time()
        if self.verb:
            print("{0:0.2f} seconds computing vector differences".format(sec2 - sec))

    # ----------------------------------------------------------------------------------------------

    #def return_neigh_vector_diffs(self, i, j):
        """Return the vector difference between points i and j.

        Args:
            i and j, indices of the two points

        Returns:
            self.X[j] - self.X[i]
        """
    #    return self.neigh_vector_diffs[self.nind_mat[i, j]]

    # ----------------------------------------------------------------------------------------------

    def compute_common_neighs(self):
        """Compute the common number of neighbours between couple of points (i,j) such that j is\
        in the neighbourhod of i. The numbers are stored in a scipy sparse csr_matrix format.

        Args:

        Returns:

        """

        # compute neighbour indices
        if self.nind_list is None:
            self.compute_neigh_indices()

        if self.verb:
            print("Computation of the numbers of common neighbours started")

        sec = time.time()
        self.common_neighs = cgr.return_common_neighs(
            self.kstar, self.dist_indices, self.nind_list
        )
        sec2 = time.time()
        if self.verb:
            print("{0:0.2f} seconds to carry out the computation.".format(sec2 - sec))

    # ----------------------------------------------------------------------------------------------

    def compute_density_kstarNN(self):
        """Computes the density of each point using a simple kNN estimator with an optimal choice of k.
        Args:
            Dthr: Likelihood ratio parameter used to compute optimal k, the value of Dthr=23.92 corresponds
            to a p-value of 1e-6.
        Returns:
        """

        if self.kstar is None:
            self.compute_kstar()

        if self.verb:
            print("kstar-NN density estimation started")

        dc = np.zeros(self.N, dtype=float)
        log_den = np.zeros(self.N, dtype=float)
        log_den_err = np.zeros(self.N, dtype=float)
        prefactor = np.exp(
            self.intrinsic_dim / 2.0 * np.log(np.pi)
            - gammaln((self.intrinsic_dim + 2) / 2)
        )

        log_den_min = 9.9e300

        for i in range(self.N):
            k = self.kstar[i]
            dc[i] = self.distances[i, k]
            log_den[i] = np.log(k) - np.log(prefactor)

            rk = self.distances[i, k]

            log_den[i] -= self.intrinsic_dim * np.log(rk)

            log_den_err[i] = 1.0 / np.sqrt(k)

            if log_den[i] < log_den_min:
                log_den_min = log_den[i]

            # Normalise density

        log_den -= np.log(self.N)

        self.log_den = log_den
        self.log_den_err = log_den_err
        self.dc = dc

        if self.verb:
            print("k-NN density estimation finished")

    # ----------------------------------------------------------------------------------------------

    def compute_density_kpeaks(self):
        if self.kstar is None:
            self.compute_kstar()

        if self.verb:
            print("Density estimation for k-peaks clustering started")

        dc = np.zeros(self.N, dtype=float)
        log_den = np.zeros(self.N, dtype=float)
        log_den_err = np.zeros(self.N, dtype=float)
        log_den_min = 9.9e300

        for i in range(self.N):
            k = self.kstar[i]
            dc[i] = self.distances[i, k]
            log_den[i] = k
            log_den_err[i] = 0
            for j in range(1, k):
                jj = self.dist_indices[i, j]
                log_den_err[i] = log_den_err[i] + (self.kstar[jj] - k) ** 2
            log_den_err[i] = np.sqrt(log_den_err[i] / k)

            if log_den[i] < log_den_min:
                log_den_min = log_den[i]

            # Normalise density

        self.log_den = log_den
        self.log_den_err = log_den_err
        self.dc = dc

        if self.verb:
            print("k-peaks density estimation finished")

    # ----------------------------------------------------------------------------------------------

    def return_density_Gaussian_kde(self, Y_sample=None, smoothing_parameter=None, adaptive=False, return_only_gradient=False):
        """Returns the logdensity of of each point in Y_sample using a Gaussian kernel density estimator based on the coordinates self.X.


        TODO: improve normalisation of gaussians on periodic range (currently normalisation of Gaussian in range (-inf,inf) 
        instead of range [-period/2,period/2])

        Args:
            Y_sample (np.ndarray(float)): the points at which the Gaussian kernel density should be evaluated. The default is self.X
            smoothing_parameter (float or np.ndarray(float)): default is given by Scott's Rule of Thumb ( self.N**(-1./(self.dims+4)) )
            adaptive (bool): if set to 'True', bandwidth is set to half the distance of the k*-th neighbour of a point, k* beingselected adaptively
            return_only_gradient (bool): if set to 'True', only returns the gradient of the logdensity

        Returns:

        """
        sec = time.time()

        # assign default sample dataset
        if Y_sample is None:
            Y_sample = self.X
            print("Selected as sample database self.X")

        assert Y_sample.shape[1] == self.dims, "The sample has dimension {} instead of required {}".format(Y_sample.shape[1],self.dims)

        # check right periodicity of Y_sample
        if self.period is not None:
            for dim in range(self.dims):
                assert np.max(Y_sample[:,dim]) <= self.period[dim] and np.min(Y_sample[:,dim]) >= 0. , "Periodic coordinates must be in range [0,period]"



        # manage smoothing parameter
        if adaptive is True:
            # check for conglicts
            if smoothing_parameter is not None:
                raise ValueError(
                    "'Either 'smoothing_parameter' must be 'None' or 'adaptive' set to 'False'"
                    )
            else:
                if self.dc is None:
                    self.compute_kstar()
                    # AAAAAAA : this is not sufficient to compute dc, needs to be implemented. At the moment
                    assert self.dc is not None, "Current Implementation of compute_kstar() does not compute dc. Use compute_density_kstarNN before in order to select adaptive bandwidth"
                smoothing_parameter = self.dc/2
                if self.verb:
                    print("Selected an adaptive smoothing parameter as dc[i]/2 for each point")
        if smoothing_parameter is not None:
            # if is np.ndarray check shape
            if isinstance(smoothing_parameter, np.ndarray):
                assert smoothing_parameter.shape == (self.N,), "smoothing_parameter should have size ({},)".format(self.N)
            # if scalar make array
            elif isinstance(smoothing_parameter, (int, float)):
                smoothing_parameter = np.full((self.N,), fill_value=smoothing_parameter, dtype=float)
            else:
                raise ValueError(
                    "'smoothing_parameter' must be either a float scalar or a numpy array of floats of shape ({},)".format(
                        self.N
                    )
                )
        else:
            # assign a smoothing parameter according to Scott's Rule of Thumb
            smoothing_parameter = self.N**(-1./(self.dims+4))
            if self.verb:
                print("Selected a smoothing parameter according to Scott's Rule of Thumb: h = {}".format(smoothing_parameter))
        
        if self.verb:
            print("Gaussian kernel density estimation started")

        if self.period is None:
            if return_only_gradient is False:
                density = cd._return_Gaussian_kde(self.X,Y_sample,smoothing_parameter)
            else:
                gradients = cd._return_gradient_Gaussian_kde(self.X,Y_sample,smoothing_parameter)
        else:
            if return_only_gradient is False:
                density = cd._return_Gaussian_kde_periodic(self.X,Y_sample,smoothing_parameter,self.period)
            else:
                gradients = cd._return_gradient_Gaussian_kde_periodic(self.X,Y_sample,smoothing_parameter,self.period)

        sec2 = time.time()
        if self.verb:
            print("{0:0.2f} seconds estimating Gaussian kernel density".format(sec2 - sec))

        if return_only_gradient is False:
            return np.log(density)
        else:
            return gradients

    # ----------------------------------------------------------------------------------------------

    def compute_density_corr_kstarNN(self):
        if self.common_neighs is None:
            self.compute_common_neighs()

        if self.verb:
            print("kstar-NN density estimation started")

        dc = np.zeros(self.N, dtype=float)
        log_den = np.zeros(self.N, dtype=float)
        #        log_den_err = np.zeros(self.N, dtype=float)
        prefactor = np.exp(
            self.intrinsic_dim / 2.0 * np.log(np.pi)
            - gammaln((self.intrinsic_dim + 2) / 2)
        )

        log_den_min = 9.9e300

        for i in range(self.N):
            k = self.kstar[i]
            dc[i] = self.distances[i, k]
            log_den[i] = np.log(k) - np.log(prefactor)

            rk = self.distances[i, k]

            log_den[i] -= self.intrinsic_dim * np.log(rk)

            #            log_den_err[i] = 1.0 / np.sqrt(k)

            if log_den[i] < log_den_min:
                log_den_min = log_den[i]

            # Normalise density

        log_den -= np.log(self.N)
        self.log_den = log_den
        self.dc = dc

        # compute error
        A = sparse.lil_matrix((self.N, self.N), dtype=np.float_)

        for nspar, indices in enumerate(self.nind_list):
            i = indices[0]
            j = indices[1]
            # A[i,j] = self.common_neighs[nspar]*1./self.kstar[i]
            A[i, j] = self.common_neighs[nspar] / 2.0
        #            if A[i,j] == 0:
        #                tmp = self.common_neighs[nspar]*2./self.N
        #                A[ i, j ] = tmp
        #                A[ j, i ] = tmp

        A = sparse.lil_matrix(A + A.transpose())

        A.setdiag(self.kstar)

        self.A = A.todense()
        self.B = slin.pinvh(self.A)
        self.log_den_err = np.sqrt(np.diag(self.B))

        if self.verb:
            print("k-NN density estimation finished")

    # ----------------------------------------------------------------------------------------------

    def compute_density_dF_PAk(self):

        # check for deltaFij
        if self.kstar is None:
            self.compute_kstar()

        # check for deltaFij
        if self.Fij_array is None:
            self.compute_deltaFs_grads_semisum()

        if self.verb:
            print("dF_PAk density estimation started")
            sec = time.time()

        dc = np.zeros(self.N, dtype=float)
        log_den = np.zeros(self.N, dtype=float)
        log_den_err = np.zeros(self.N, dtype=float)
        prefactor = np.exp(
            self.intrinsic_dim / 2.0 * np.log(np.pi)
            - gammaln((self.intrinsic_dim + 2) / 2)
        )

        log_den_min = 9.9e300

        for i in range(self.N):
            k = int(self.kstar[i])
            dc[i] = self.distances[i, k]
            log_den[i] = np.log(k) - np.log(prefactor)

            # log_den_err[i] = np.sqrt((4 * k + 2) / (k * (k - 1)))
            corrected_rk = 0.0
            Fijs = self.Fij_array[self.nind_iptr[i] : self.nind_iptr[i + 1]]

            for j in range(1, k):
                Fij = Fijs[j - 1]
                rjjm1 = (
                    self.distances[i, j] ** self.intrinsic_dim
                    - self.distances[i, j - 1] ** self.intrinsic_dim
                )

                corrected_rk += rjjm1 * np.exp(Fij)  # * (1+Fij)

            log_den[i] -= np.log(corrected_rk)

            if log_den[i] < log_den_min:
                log_den_min = log_den[i]

                # Normalise density
        log_den -= np.log(self.N)

        self.log_den = log_den
        self.log_den_err = 1.0 / np.sqrt(self.kstar)
        self.dc = dc

        sec2 = time.time()
        if self.verb:
            print("{0:0.2f} seconds for dF_PAk density estimation".format(sec2 - sec))

    # ----------------------------------------------------------------------------------------------

    def compute_density_PAk(self):

        # compute optimal k
        if self.kstar is None:
            self.compute_kstar()

        if self.verb:
            print("PAk density estimation started")

        dc = np.zeros(self.N, dtype=float)
        log_den = np.zeros(self.N, dtype=float)
        log_den_err = np.zeros(self.N, dtype=float)
        prefactor = np.exp(
            self.intrinsic_dim / 2.0 * np.log(np.pi)
            - gammaln((self.intrinsic_dim + 2.0) / 2.0)
        )
        log_den_min = 9.9e300

        sec = time.time()
        for i in range(self.N):
            vi = np.zeros(self.maxk, dtype=float)
            dc[i] = self.distances[i, self.kstar[i]]
            rr = np.log(self.kstar[i]) - (
                np.log(prefactor)
                + self.intrinsic_dim * np.log(self.distances[i, self.kstar[i]])
            )
            knn = 0
            for j in range(self.kstar[i]):
                # to avoid easy overflow
                vi[j] = prefactor * (
                    pow(self.distances[i, j + 1], self.intrinsic_dim)
                    - pow(self.distances[i, j], self.intrinsic_dim)
                )

                if vi[j] < 1.0e-300:
                    knn = 1
                    break
            if knn == 0:
                log_den[i] = cml._nrmaxl(rr, self.kstar[i], vi, self.maxk)
            else:
                log_den[i] = rr
            if log_den[i] < log_den_min:
                log_den_min = log_den[i]

            log_den_err[i] = np.sqrt(
                (4 * self.kstar[i] + 2) / (self.kstar[i] * (self.kstar[i] - 1))
            )

        sec2 = time.time()

        if self.verb:
            print(
                "{0:0.2f} seconds optimizing the likelihood for all the points".format(
                    sec2 - sec
                )
            )

        # Normalise density
        log_den -= np.log(self.N)

        self.log_den = log_den
        self.log_den_err = log_den_err
        self.dc = dc

        if self.verb:
            print("PAk density estimation finished")

    # ----------------------------------------------------------------------------------------------

    def compute_density_gCorr(self, use_variance=True,comp_err=True):
        # TODO: matrix A should be in sparse format!

        # compute changes in free energy
        if self.Fij_array is None:
            self.compute_deltaFs_grads_semisum()

        if self.verb:
            print("gCorr density estimation started")
            sec = time.time()

        # compute adjacency matrix and cumulative changes
        A = sparse.lil_matrix((self.N, self.N), dtype=np.float_)

        supp_deltaF = sparse.lil_matrix((self.N, self.N), dtype=np.float_)

        # define redundancy factor for each A matrix entry as the geometric mean of the 2 corresponding k*
        k1 = self.kstar[self.nind_list[:, 0]]
        k2 = self.kstar[self.nind_list[:, 1]]
        redundancy = np.sqrt(k1 * k2)

        if use_variance:
            for nspar, indices in enumerate(self.nind_list):
                i = indices[0]
                j = indices[1]
                # tmp = 1.0 / self.Fij_var_array[nspar]
                tmp = 1.0 / self.Fij_var_array[nspar] / redundancy[nspar]
                A[i, j] = -tmp
                supp_deltaF[i, j] = self.Fij_array[nspar] * tmp
        else:
            for nspar, indices in enumerate(self.nind_list):
                i = indices[0]
                j = indices[1]
                # A[i, j] = -1.0
                A[i, j] = -1.0 / redundancy[nspar]
                supp_deltaF[i, j] = self.Fij_array[nspar]

        A = sparse.lil_matrix(A + A.transpose())

        diag = np.array(-A.sum(axis=1)).reshape((self.N,))

        A.setdiag(diag)

        # print("Diag = {}".format(diag))

        deltaFcum = np.array(supp_deltaF.sum(axis=0)).reshape((self.N,)) - np.array(
            supp_deltaF.sum(axis=1)
        ).reshape((self.N,))

        sec2 = time.time()
        if self.verb:
            print("{0:0.2f} seconds to fill sparse matrix".format(sec2 - sec))

        log_den = sparse.linalg.spsolve(A.tocsr(), deltaFcum)

        if self.verb:
            print("{0:0.2f} seconds to solve linear system".format(time.time() - sec2))
        sec2 = time.time()

        self.log_den = log_den
        # self.log_den_err = np.sqrt((sparse.linalg.inv(A.tocsc())).diagonal())
        
        if comp_err is True:
            self.A = A.todense()
            self.B = slin.pinvh(self.A)
            # self.B = slin.inv(self.A)
            self.log_den_err = np.sqrt(np.diag(self.B))

        if self.verb:
            print("{0:0.2f} seconds inverting A matrix".format(time.time() - sec2))
        sec2 = time.time()

        # self.log_den_err = np.sqrt(np.diag(slin.pinvh(A.todense())))
        # self.log_den_err = np.sqrt(diag/np.array(np.sum(np.square(A.todense()),axis=1)).reshape(self.N,))

        sec2 = time.time()
        if self.verb:
            print("{0:0.2f} seconds for gCorr density estimation".format(sec2 - sec))

    # ----------------------------------------------------------------------------------------------

    def compute_density_kstarNN_gCorr(
        self, alpha=1.0, gauss_approx=False, Fij_type="grad"
    ):
        """
        finds the minimum of the
        """
        from duly.utils_.mlmax_pytorch import maximise

        # Fij_types: 'grad', 'zero', 'PAk'
        # TODO: we need to implement a gCorr term with the deltaFijs equal to zero.
        # Implementation is obsolete, we should use same ad dF_PAk_gCorr

        # compute optimal k
        if self.kstar is None:
            self.compute_kstar()

        if Fij_type == "zero":
            # set changes in free energy to zero
            raise NotImplementedError("still not implemented")
            

        elif Fij_type == "grad":
            # compute changes in free energy
            if self.Fij_array is None:
                self.compute_deltaFs_grads_semisum()

            for i in range(self.N):

                self.Fij_list.append(
                    self.Fij_array[self.nind_iptr[i] : self.nind_iptr[i + 1]]
                )
                self.Fij_var_list.append(
                    self.Fij_var_array[self.nind_iptr[i] : self.nind_iptr[i + 1]]
                )

        else:
            raise ValueError("please select a valid Fij type")

        if gauss_approx:
            raise NotImplementedError(
                "Gaussian approximation not yet implemented (MATTEO DO IT)"
            )

        else:
            # compute Vis
            prefactor = np.exp(
                self.intrinsic_dim / 2.0 * np.log(np.pi)
                - gammaln((self.intrinsic_dim + 2) / 2)
            )

            dc = np.array([self.distances[i, self.kstar[i]] for i in range(self.N)])

            Vis = prefactor * (dc ** self.intrinsic_dim)

            # get good initial conditions for the optimisation
            Fis = np.array(
                [np.log(self.kstar[i]) - np.log(Vis[i]) for i in range(self.N)]
            )

            # optimise the likelihood using pytorch
            l_, log_den = maximise(
                Fis, self.kstar, Vis, self.dist_indices, Fij_list, Fij_var_list, alpha
            )

        # normalise density
        log_den -= np.log(self.N)

        self.log_den = log_den

    # ----------------------------------------------------------------------------------------------

    def compute_density_dF_PAk_gCorr(self, use_variance=True, alpha=1.0, comp_err=True):

        # check for deltaFij
        if self.Fij_array is None:
            self.compute_deltaFs_grads_semisum()

        if self.verb:
            print("dF_PAk_gCorr density estimation started")
            sec = time.time()

        dc = np.zeros(self.N, dtype=float)
        corrected_vols = np.zeros(self.N, dtype=float)
        log_den = np.zeros(self.N, dtype=float)
        log_den_err = np.zeros(self.N, dtype=float)
        prefactor = np.exp(
            self.intrinsic_dim / 2.0 * np.log(np.pi)
            - gammaln((self.intrinsic_dim + 2) / 2)
        )

        log_den_min = 9.9e300

        for i in range(self.N):
            k = int(self.kstar[i])
            dc[i] = self.distances[i, k]
            Fijs = self.Fij_array[self.nind_iptr[i] : self.nind_iptr[i + 1]]

            for j in range(1, k):
                Fij = Fijs[j - 1]
                rjjm1 = (
                    self.distances[i, j] ** self.intrinsic_dim
                    - self.distances[i, j - 1] ** self.intrinsic_dim
                )

                corrected_vols[i] += rjjm1 * np.exp(Fij)  # * (1+Fij)

        corrected_vols *= prefactor * self.N

        self.dc = dc

        # compute adjacency matrix and cumulative changes
        A = sparse.lil_matrix((self.N, self.N), dtype=np.float_)

        supp_deltaF = sparse.lil_matrix((self.N, self.N), dtype=np.float_)

        # define redundancy factor for each A matrix entry as the geometric mean of the 2 corresponding k*
        k1 = self.kstar[self.nind_list[:, 0]]
        k2 = self.kstar[self.nind_list[:, 1]]
        redundancy = np.sqrt(k1 * k2)

        if use_variance:
            for nspar, indices in enumerate(self.nind_list):
                i = indices[0]
                j = indices[1]
                tmp = 1.0 / self.Fij_var_array[nspar] / redundancy[nspar]
                A[i, j] = -tmp
                supp_deltaF[i, j] = self.Fij_array[nspar] * tmp
        else:
            for nspar, indices in enumerate(self.nind_list):
                i = indices[0]
                j = indices[1]
                A[i, j] = -1.0 / redundancy[nspar]
                supp_deltaF[i, j] = self.Fij_array[nspar] / redundancy[nspar]

        A = alpha * sparse.lil_matrix(A + A.transpose())

        diag = np.array(-A.sum(axis=1)).reshape((self.N,)) + (1.0 - alpha) * self.kstar

        #        print("Diag = {}".format(diag))

        A.setdiag(diag)

        deltaFcum = alpha * (
            np.array(supp_deltaF.sum(axis=0)).reshape((self.N,))
            - np.array(supp_deltaF.sum(axis=1)).reshape((self.N,))
        ) + (1.0 - alpha) * (self.kstar * (np.log(self.kstar / corrected_vols)))

        log_den = sparse.linalg.spsolve(A.tocsr(), deltaFcum)

        self.log_den = log_den

        if comp_err is True:
            self.A = A.todense()
            self.B = slin.pinvh(self.A)
            self.log_den_err = np.sqrt(np.diag(self.B))

        sec2 = time.time()
        if self.verb:
            print(
                "{0:0.2f} seconds for dF_PAk_gCorr density estimation".format(
                    sec2 - sec
                )
            )

    # ----------------------------------------------------------------------------------------------

    def compute_density_PAk_gCorr(
        self, gauss_approx=True, alpha=1.0, log_den_PAk=None, log_den_PAk_err=None, comp_err=True
    ):
        """
        finds the maximum likelihood solution of PAk likelihood + gCorr likelihood with deltaFijs
        computed using the gradients
        """
        # TODO: we need to impement the deltaFijs to be computed as a*l (as in PAk)

        # compute changes in free energy
        if self.Fij_array is None:
            self.compute_deltaFs_grads_semisum()

        if self.verb:
            print("PAk_gCorr density estimation started")
            sec = time.time()

        dc = np.empty(self.N, dtype=float)
        log_den = np.empty(self.N, dtype=float)
        log_den_err = np.zeros(self.N, dtype=float)
        prefactor = np.exp(
            self.intrinsic_dim / 2.0 * np.log(np.pi)
            - gammaln((self.intrinsic_dim + 2) / 2)
        )
        log_den_min = 9.9e300
        vij_list = []
        Fij_list = []
        Fij_var_list = []

        if gauss_approx is True:
            if self.verb:
                print("Maximising likelihood in Gaussian approximation")

            if log_den_PAk is not None and log_den_PAk_err is not None:
                self.log_den = log_den_PAk
                self.log_den_err = log_den_PAk_err

            else:
                self.compute_density_PAk()

            # compute adjacency matrix and cumulative changes
            A = sparse.lil_matrix((self.N, self.N), dtype=np.float_)

            supp_deltaF = sparse.lil_matrix((self.N, self.N), dtype=np.float_)

            # define redundancy factor for each A matrix entry as the geometric mean of the 2 corresponding k*
            k1 = self.kstar[self.nind_list[:, 0]]
            k2 = self.kstar[self.nind_list[:, 1]]
            redundancy = np.sqrt(k1 * k2)

            for nspar, indices in enumerate(self.nind_list):
                i = indices[0]
                j = indices[1]
                # tmp = 1.0 / self.Fij_var_array[nspar]
                tmp = 1.0 / self.Fij_var_array[nspar] / redundancy[nspar]
                A[i, j] = -tmp
                supp_deltaF[i, j] = self.Fij_array[nspar] * tmp

            A = alpha * sparse.lil_matrix(A + A.transpose())

            diag = (
                np.array(-A.sum(axis=1)).reshape((self.N,))
                + (1.0 - alpha) / self.log_den_err ** 2
            )

            A.setdiag(diag)

            deltaFcum = (
                alpha
                * (
                    np.array(supp_deltaF.sum(axis=0)).reshape((self.N,))
                    - np.array(supp_deltaF.sum(axis=1)).reshape((self.N,))
                )
                + (1.0 - alpha) * self.log_den / self.log_den_err ** 2
            )

            sec2 = time.time()
            if self.verb:
                print("{0:0.2f} seconds to fill sparse matrix".format(sec2 - sec))

            log_den = sparse.linalg.spsolve(A.tocsr(), deltaFcum)


            if self.verb:
                print("{0:0.2f} seconds to solve linear system".format(time.time() - sec2))
            sec2 = time.time()

            self.log_den = log_den

            if comp_err is True:
                self.A = A.todense()
                self.B = slin.pinvh(self.A)
                # self.B = slin.inv(self.A)
                self.log_den_err = np.sqrt(np.diag(self.B))

            if self.verb:
                print("{0:0.2f} seconds inverting A matrix".format(time.time() - sec2))
            sec2 = time.time()

            # self.log_den_err = np.sqrt(diag/(np.array(np.sum(np.square(A.todense()),axis=1)).reshape(self.N,)))

        else:
            if self.verb:
                print("Solving via SGD")
            from duly.utils_.mlmax_pytorch import maximise_wPAk

            for i in range(self.N):

                Fij_list.append(
                    self.Fij_array[self.nind_iptr[i] : self.nind_iptr[i + 1]]
                )
                Fij_var_list.append(
                    self.Fij_var_array[self.nind_iptr[i] : self.nind_iptr[i + 1]]
                )

                dc[i] = self.distances[i, self.kstar[i]]
                rr = np.log(self.kstar[i]) - (
                    np.log(prefactor)
                    + self.intrinsic_dim * np.log(self.distances[i, self.kstar[i]])
                )
                log_den[i] = rr
                vj = np.zeros(self.kstar[i])
                for j in range(self.kstar[i]):
                    vj[j] = prefactor * (
                        pow(self.distances[i, j + 1], self.intrinsic_dim)
                        - pow(self.distances[i, j], self.intrinsic_dim)
                    )

                vij_list.append(vj)

            l_, log_den = maximise_wPAk(
                log_den,
                self.kstar,
                vij_list,
                self.dist_indices,
                Fij_list,
                Fij_var_list,
                alpha,
            )
            log_den -= np.log(self.N)

        self.log_den = log_den

        sec2 = time.time()
        if self.verb:
            print(
                "{0:0.2f} seconds for PAk_gCorr density estimation".format(sec2 - sec)
            )

    # ----------------------------------------------------------------------------------------------
   
    def compute_grads(self, comp_covmat=False):
        """Compute the gradient of the log density each point using k* nearest neighbors.
        The gradient is estimated via a linear expansion of the density propagated to the log-density.

        Args:

        Returns:

        MODIFICARE QUI E ANCHE NEGLI ATTRIBUTI

        """
        # compute optimal k
        if self.kstar is None:
            self.compute_kstar()

        if self.verb:
            print("Estimation of the density gradient started")

        sec = time.time()
        if comp_covmat is False:
            # self.grads, self.grads_var = cgr.return_grads_and_var_from_coords(self.X, self.dist_indices, self.kstar, self.intrinsic_dim)
            self.grads, self.grads_var = cgr.return_grads_and_var_from_nnvecdiffs(
                self.neigh_vector_diffs,
                self.nind_list,
                self.nind_iptr,
                self.kstar,
                self.intrinsic_dim,
            )
        else:
            # self.grads, self.grads_var = cgr.return_grads_and_covmat_from_coords(self.X, self.dist_indices, self.kstar, self.intrinsic_dim)
            self.grads, self.grads_var = cgr.return_grads_and_covmat_from_nnvecdiffs(
                self.neigh_vector_diffs,
                self.nind_list,
                self.nind_iptr,
                self.kstar,
                self.intrinsic_dim,
            )
            self.check_grads_covmat = True
        sec2 = time.time()
        if self.verb:
            print("{0:0.2f} seconds computing gradients".format(sec2 - sec))

    # ----------------------------------------------------------------------------------------------

    def compute_deltaFs_grads_semisum(self, chi="auto"):
        """Compute deviations deltaFij to standard kNN log-densities at point j as seen from point i using\
            a linear expansion with as slope the semisum of the average gradient of the log-density over the neighbourhood of points i and j. \
            The parameter chi is used in the estimation of the squared error of the deltaFij as 1/4*(E_i^2+E_j^2+2*E_i*E_j*chi), \
            where E_i is the error on the estimate of grad_i*DeltaX_ij.

        Args:
            chi: the Pearson correlation coefficient between the estimates of the gradient in i and j. Can take a numerical value between 0 and 1.\
                The option 'auto' takes a geometrical estimate of chi based on AAAAAAAAA

        Returns:

        """

        # check or compute vector_diffs
        if self.neigh_vector_diffs is None:
            self.compute_neigh_vector_diffs()

        # check or compute gradients and their covariance matrices
        if self.grads is None:
            self.compute_grads(comp_covmat=True)

        elif self.check_grads_covmat == False:
            self.compute_grads(comp_covmat=True)

        if self.verb:
            print(
                "Estimation of the gradient semisum (linear) corrections deltaFij to the log-density started"
            )
        sec = time.time()

        Fij_array = np.zeros(self.nspar)
        Fij_var_array = np.zeros(self.nspar)

        k1 = self.kstar[self.nind_list[:, 0]]
        k2 = self.kstar[self.nind_list[:, 1]]
        g1 = self.grads[self.nind_list[:, 0]]
        g2 = self.grads[self.nind_list[:, 1]]
        g_var1 = self.grads_var[self.nind_list[:, 0]]
        g_var2 = self.grads_var[self.nind_list[:, 1]]

        if chi == "auto":
            # check or compute common_neighs
            if self.common_neighs is None:
                self.compute_common_neighs()

            chi = self.common_neighs / (k1 + k2 - self.common_neighs)

        else:
            assert chi >= 0.0 and chi <= 1.0, "Invalid value for argument 'chi'"

        Fij_array = 0.5 * np.einsum("ij, ij -> i", g1 + g2, self.neigh_vector_diffs)
        vari = np.einsum(
            "ij, ij -> i",
            self.neigh_vector_diffs,
            np.einsum("ijk, ik -> ij", g_var1, self.neigh_vector_diffs),
        )
        varj = np.einsum(
            "ij, ij -> i",
            self.neigh_vector_diffs,
            np.einsum("ijk, ik -> ij", g_var2, self.neigh_vector_diffs),
        )
        Fij_var_array = 0.25 * (vari + varj + 2 * chi * np.sqrt(vari * varj))

        sec2 = time.time()
        if self.verb:
            print("{0:0.2f} seconds computing gradient corrections".format(sec2 - sec))

        self.Fij_array = Fij_array
        self.Fij_var_array = Fij_var_array

    # ----------------------------------------------------------------------------------------------

    def return_entropy(self):
        """Compute a very rough estimate of the entropy of the data distribution
        as the average negative log probability.

        Returns:
            H (float): the estimate entropy of the distribution

        """
        assert self.log_den is not None

        H = -np.mean(self.log_den)

        return H


    def return_interpolated_density_kNN(self, X_new, k):

        cross_distances, cross_dist_indices = compute_cross_nn_distances(
            X_new, self.X, self.maxk, self.metric, self.p, self.period
        )

        kstar = np.ones(X_new.shape[0], dtype=int) * k

        log_den, log_den_err, dc = return_density_kstarNN(
            cross_distances, self.intrinsic_dim, kstar
        )

        return log_den, log_den_err

    def return_interpolated_density_kstarNN(self, X_new, Dthr=23.92812698):

        cross_distances, cross_dist_indices = compute_cross_nn_distances(
            X_new, self.X, self.maxk, self.metric, self.p, self.period
        )

        kstar = cd._compute_kstar_interp(
            self.intrinsic_dim,
            X_new.shape[0],
            self.maxk,
            Dthr,
            cross_dist_indices,
            cross_distances,
            self.distances,
        )

        log_den, log_den_err, dc = return_density_kstarNN(
            cross_distances, self.intrinsic_dim, kstar
        )

        # correction for interpolation
        log_den = log_den - np.log(kstar) + np.log(kstar - 1)
        log_den_err = log_den_err * np.sqrt(kstar / (kstar - 1))

        return log_den, log_den_err

    def return_interpolated_density_PAk(self, X_new, Dthr=23.92812698):

        assert self.intrinsic_dim is not None
        assert self.X is not None

        cross_distances, cross_dist_indices = compute_cross_nn_distances(
            X_new, self.X, self.maxk, self.metric, self.p, self.period
        )

        kstar = cd._compute_kstar_interp(
            self.intrinsic_dim,
            X_new.shape[0],
            self.maxk,
            Dthr,
            cross_dist_indices,
            cross_distances,
            self.distances,
        )

        log_den, log_den_err, dc = return_density_PAk(
            cross_distances, self.intrinsic_dim, kstar, self.maxk
        )

        # correction for interpolation missing!

        return log_den, log_den_err


if __name__ == "__main__":
    X = np.random.uniform(size=(1000, 2))

    de = DensityEstimation(coordinates=X)

    de.compute_distances(maxk=300)

    de.compute_id_2NN()

    de.compute_density_PAk_optimised()

    # de.compute_grads()

    print(de.log_den)
