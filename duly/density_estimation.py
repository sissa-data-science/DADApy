import time
import multiprocessing

import numpy as np
from scipy.special import gammaln
from scipy import sparse

from duly.id_estimation import IdEstimation
from duly.utils_.mlmax import MLmax_gPAk, MLmax_gpPAk
from duly.cython_ import cython_functions as cf

cores = multiprocessing.cpu_count()


class DensityEstimation(IdEstimation):
    """Computes the log-density and its error at each point and other properties.

    Inherits from class IdEstimation. Can estimate the optimal number k* of neighbors for each points. \
    Can compute the log-density and its error at each point choosing among various kNN-based methods. \
    Can return an estimate of the gradient of the log-density at each point and an estimate of the error on each component. \
    Can return an estimate of the linear deviation from constant density at each point and an estimate of the error on each component. \

    Attributes:
        kstar (np.array(float)): array containing the chosen number k* of neighbors for each of the Nele points
        array_vector_diffs (np.ndarray(float), optional): stores vector differences from each point to its k* nearest neighbors. Accessed by the method return_vector_diffs(i,j) for each j in the neighbourhood of i
        common_neighs (scipy.sparse.csr_matrix(float), optional): stored as a sparse symmetric matrix of size Nele x Nele. Entry (i,j) gives the common number of neighbours between points i and j. Such value is reliable only if j is in the neighbourhood of i or vice versa.
        dc (np.array(float), optional): array containing the distance of the k*th neighbor from each of the Nele points
        Rho (np.array(float), optional): array containing the Nele log-densities
        Rho_err (np.array(float), optional): array containing the Nele errors on the Rho
        grads (np.ndarray(float), optional): for each line i contains the gradient components estimated from from point i 
        grads_var (np.ndarray(float), optional): for each line i contains the estimated variance of the gradient components at point i
        check_grads_covmat (bool, optional): it is flagged "True" when grads_var contains the variance-covariance matrices of the gradients.
        Fij_list (list(np.array(float)), optional): list of Nele arrays containing for each i the k* estimates of deltaF_ij computed from point i
        Fij_var_list (list(np.array(float)), optional): list of Nele arrays containing the squared errors on the deltaF_ij's


    """

    def __init__(self, coordinates=None, distances=None, maxk=None, verbose=False, njobs=cores):
        super().__init__(coordinates=coordinates, distances=distances, maxk=maxk, verbose=verbose,
                         njobs=njobs)

        self.kstar=None
        self.neigh_vector_diffs=None
        self.nind_list=None
        self.nind_iptr=None
        self.nind_mat=None
        self.common_neighs=None
        self.dc=None
        self.Rho=None
        self.Rho_err=None
        self.grads=None
        self.grads_var=None
        self.Fij_list=None
        self.Fij_var_list=None
        self.Fij_array=None
        self.Fij_var_array=None

    # ----------------------------------------------------------------------------------------------

    def compute_density_kNN(self, k=3):
        """Compute the density of of each point using a simple kNN estimator

        Args:
            k: number of neighbours used to compute the density

        Returns:

        """
        assert (self.id_selected is not None)

        if self.verb: print('k-NN density estimation started (k={})'.format(k))

        kstar = np.full(self.Nele, k, dtype=int)
        dc = np.zeros(self.Nele, dtype=float)
        Rho = np.zeros(self.Nele, dtype=float)
        Rho_err = np.zeros(self.Nele, dtype=float)
        prefactor = np.exp(
            self.id_selected / 2. * np.log(np.pi) - gammaln((self.id_selected + 2) / 2))
        Rho_min = 9.9E300

        for i in range(self.Nele):
            dc[i] = self.distances[i, k]
            Rho[i] = np.log(kstar[i]) - (
                    np.log(prefactor) + self.id_selected * np.log(self.distances[i, kstar[i]]))

            Rho_err[i] = 1. / np.sqrt(k)
            if (Rho[i] < Rho_min):
                Rho_min = Rho[i]

        # Normalise density
        Rho -= np.log(self.Nele)

        self.Rho = Rho
        self.Rho_err = Rho_err
        self.dc = dc
        self.kstar = kstar

        if self.verb: print('k-NN density estimation finished')

    # ----------------------------------------------------------------------------------------------

    def compute_kstar(self, Dthr=23.92812698):
        """Computes the density of each point using a simple kNN estimator with an optimal choice of k.

        Args:
            Dthr: Likelihood ratio parameter used to compute optimal k, the value of Dthr=23.92 corresponds
            to a p-value of 1e-6.

        Returns:

        """
        if self.id_selected is None:
            self.compute_id_2NN()

        if self.verb:
            print('kstar estimation started, Dthr = {}'.format(Dthr))

        # Dthr = 23.92812698  # this threshold value corresponds to being sure within a p-value
        # of 1E-6 that the k-NN densities, do not touch unless you really know what you are doing

        # Array initialization for kstar
        kstar = np.empty(self.Nele, dtype=int)
        prefactor = np.exp(
            self.id_selected / 2. * np.log(np.pi) - gammaln((self.id_selected + 2.) / 2.))

        sec = time.time()
        for i in range(self.Nele):
            j = 4
            dL = 0.
            while (j < self.maxk and dL < Dthr):
                ksel = j - 1
                vvi = prefactor * pow(self.distances[i, ksel], self.id_selected)
                vvj = prefactor * pow(self.distances[self.dist_indices[i, j], ksel],
                                      self.id_selected)
                dL = -2. * ksel * (np.log(vvi) + np.log(vvj) - 2. * np.log(vvi + vvj) + np.log(4))
                j = j + 1
            kstar[i] = j - 2
        sec2 = time.time()
        if self.verb:
            print( "{0:0.2f} seconds finding the optimal k for all the points".format(sec2 - sec) )

        self.kstar = kstar

    # ----------------------------------------------------------------------------------------------

    def compute_neigh_indices(self):
        """Compute the indices of all the couples [i,j] such that j is a neighbour of i up to the k*-th nearest (excluded).
        The couples of indices are stored in a numpy ndarray of rank 2 and secondary dimension = 2.
        The index of the corresponding AAAAAAAAAAAAA make indpointer which is a np.array of length Nele which indicates for each i the starting index of the corresponding [i,.] subarray.

        """
        # compute optimal k
        if self.kstar is None:
            self.compute_kstar()

        if self.verb:
            print('Computation of the neighbour indices started')

        sec = time.time()

        #self.get_vector_diffs = sparse.csr_matrix((self.Nele, self.Nele),dtype=np.int_)
        self.nind_list , self.nind_iptr = cf.return_neigh_ind(self.dist_indices, self.kstar)

        sec2 = time.time()
        if self.verb: print(
            "{0:0.2f} seconds computing vector differences".format(sec2 - sec))

    # ----------------------------------------------------------------------------------------------

    def compute_neigh_vector_diffs(self):
        """Compute the vector differences from each point to its k* nearest neighbors.
        The resulting vectors are stored in a numpy ndarray of rank 2 and secondary dimension = dims.
        The index of  scipy sparse csr_matrix format.

        """
        # compute neighbour indices
        if self.nind_list is None: self.compute_neigh_indices()

        if self.verb: print('Computation of the vector differences started')
        sec = time.time()

        #self.get_vector_diffs = sparse.csr_matrix((self.Nele, self.Nele),dtype=np.int_)
        self.neigh_vector_diffs = cf.return_neigh_vector_diffs(self.X, self.nind_list)

        sec2 = time.time()
        if self.verb: print(
            "{0:0.2f} seconds computing vector differences".format(sec2 - sec))

    # ----------------------------------------------------------------------------------------------

    def return_neigh_vector_diffs(self, i, j):
        """Return the vector difference between points i and j.

        Args:
            i and j, indices of the two points

        Returns:
            self.X[j] - self.X[i]
        """ 
        return self.neigh_vector_diffs[self.nind_mat[i,j]]

    # ----------------------------------------------------------------------------------------------

    def compute_common_neighs(self):
        """Compute the common number of neighbours between couple of points (i,j) such that j is\
        in the neighbourhood of i. The numbers are stored in a scipy sparse csr_matrix format.

        Args:

        Returns:

        """

        # compute neighbour indices
        if self.nind_list is None: self.compute_neigh_indices()

        if self.verb: print('Computation of the numbers of common neighbours started')

        sec = time.time()
        self.common_neighs = cf.return_common_neighs(self.kstar, self.dist_indices, self.nind_list)
        sec2 = time.time()
        if self.verb: print(
            "{0:0.2f} seconds to carry out the computation.".format(sec2 - sec))

    # ----------------------------------------------------------------------------------------------

    def compute_density_kstarNN(self):
        if self.kstar is None:
            self.compute_kstar()

        if self.verb: print('kstar-NN density estimation started')

        dc = np.zeros(self.Nele, dtype=float)
        Rho = np.zeros(self.Nele, dtype=float)
        Rho_err = np.zeros(self.Nele, dtype=float)
        prefactor = np.exp(
            self.id_selected / 2. * np.log(np.pi) - gammaln((self.id_selected + 2) / 2))

        Rho_min = 9.9E300

        for i in range(self.Nele):
            k =self.kstar[i]
            dc[i] = self.distances[i, k]
            Rho[i] = np.log(k) - np.log(prefactor)

            rk = self.distances[i, k]

            Rho[i] -= self.id_selected * np.log(rk)

            Rho_err[i] = 1. / np.sqrt(k)

            if (Rho[i] < Rho_min):
                Rho_min = Rho[i]

            # Normalise density

        Rho -= np.log(self.Nele)

        self.Rho = Rho
        self.Rho_err = Rho_err
        self.dc = dc

        if self.verb: print('k-NN density estimation finished')

    # ----------------------------------------------------------------------------------------------

    def compute_density_PAk(self, method='NR'):

        # options for method:
        #   - "NR"=Newton-Raphson implemented in cython
        #   - "NM"=Nelder-Mead scipy built-in
        #   - "For"=Newton-Raphson implemented in Fortran

        # compute optimal k
        if self.kstar is None:
            self.compute_kstar()

        if self.verb:
            print('PAk density estimation started')

        dc = np.zeros(self.Nele, dtype=float)
        Rho = np.zeros(self.Nele, dtype=float)
        Rho_err = np.zeros(self.Nele, dtype=float)
        prefactor = np.exp(
            self.id_selected / 2. * np.log(np.pi) - gammaln((self.id_selected + 2.) / 2.))
        Rho_min = 9.9E300

        sec = time.time()
        for i in range(self.Nele):
            vi = np.zeros(self.maxk, dtype=float)
            dc[i] = self.distances[i, self.kstar[i]]
            rr = np.log(self.kstar[i]) - (
                    np.log(prefactor) + self.id_selected * np.log(self.distances[i, self.kstar[i]]))
            knn = 0
            for j in range(self.kstar[i]):
                # to avoid easy overflow
                vi[j] = prefactor * (pow(self.distances[i, j + 1], self.id_selected) - pow(
                    self.distances[i, j], self.id_selected))
                # distance_ratio = pow(self.distances[i, j]/self.distances[i, j + 1], self.id_selected)
                # print(distance_ratio)
                # exponent = self.id_selected*np.log(self.distances[i, j + 1]) + np.log(1-distance_ratio)
                # print(exponent)
                # vi[j] = prefactor*np.exp(exponent)
                if (vi[j] < 1.0E-300):
                    knn = 1
                    break
            if (knn == 0):
                if (method == 'NR'):
                    Rho[i] = cf._nrmaxl(rr, self.kstar[i], vi, self.maxk)
                elif (method == 'NM'):
                    from duly.utils_.mlmax import MLmax
                    Rho[i] = MLmax(rr, self.kstar[i], vi)
                else:
                    raise ValueError("Please choose a valid method")
                # Rho[i] = NR.nrmaxl(rr, kstar[i], vi, self.maxk) # OLD FORTRAN
            else:
                Rho[i] = rr
            if (Rho[i] < Rho_min):
                Rho_min = Rho[i]

            Rho_err[i] = np.sqrt((4 * self.kstar[i] + 2) / (self.kstar[i] * (self.kstar[i] - 1)))

        sec2 = time.time()
        if self.verb:
            print("{0:0.2f} seconds optimizing the likelihood for all the points".format(sec2 - sec))

        # Normalise density
        Rho -= np.log(self.Nele)

        self.Rho = Rho
        self.Rho_err = Rho_err
        self.dc = dc
        self.kstar = self.kstar

        if self.verb:
            print('PAk density estimation finished')

    # ----------------------------------------------------------------------------------------------

    def compute_density_kstarNN_gCorr(self, alpha=1., gauss_approx=False, Fij_type='grad'):
        """
        finds the minimum of the
        """
        from duly.utils_.mlmax_pytorch import maximise
        # Fij_types: 'grad', 'zero', 'PAk'
        # TODO: we need to implement a gCorr term with the deltaFijs equal to zero

        # compute optimal k
        if self.kstar is None:
            self.compute_kstar()

        if Fij_type == 'zero':
            # set changes in free energy to zero
            raise NotImplementedError("still not implemented")
            # self.Fij_list = []
            # self.Fij_var_list = []
            #
            # Fij_list = self.Fij_list
            # Fij_var_list = self.Fij_var_list

        elif Fij_type == 'grad':
            # compute changes in free energy
            if self.Fij_list is None: self.compute_deltaFs_grad()
            Fij_list = self.Fij_list
            Fij_var_list = self.Fij_var_list

        else:
            raise ValueError("please select a valid Fij type")

        if gauss_approx:
            raise NotImplementedError("Gaussian approximation not yet implemented (MATTEO DO IT)")

        else:
            # compute Vis
            prefactor = np.exp(
                self.id_selected / 2. * np.log(np.pi) - gammaln((self.id_selected + 2) / 2))

            dc = np.array([self.distances[i,self.kstar[i]] for i in range(self.Nele)])

            Vis = prefactor * (dc ** self.id_selected)

            # get good initial conditions for the optimisation
            Fis = np.array([np.log(self.kstar[i]) - np.log(Vis[i]) for i in range(self.Nele)])

            # optimise the likelihood using pytorch
            l_, Rho = maximise(Fis,self.kstar, Vis, self.dist_indices, Fij_list, Fij_var_list, alpha)

        # normalise density
        Rho -= np.log(self.Nele)

        self.Rho = Rho

    # ----------------------------------------------------------------------------------------------

    def compute_density_PAk_gCorr(self, alpha=1.):
        from duly.utils_.mlmax_pytorch import maximise_wPAk
        """
        finds the maximum likelihood solution of PAk likelihood + gCorr likelihood with deltaFijs
        computed using the gradients
        """
        # TODO: we need to impement the deltaFijs to be computed as a*l (as in PAk)

        # compute changes in free energy
        if self.Fij_array is None: self.compute_deltaFs_grads_semisum()

        dc = np.empty(self.Nele, dtype=float)
        Rho = np.empty(self.Nele, dtype=float)

        prefactor = np.exp(
            self.id_selected / 2. * np.log(np.pi) - gammaln((self.id_selected + 2) / 2))

        vij_list = []

        for i in range(self.Nele):
            
            Fij_list = self.Fij_list
            Fij_var_list = self.Fij_var_list

            Fij_list.append(self.Fij_array[self.nind_iptr[i]:self.nind_iptr[i+1]])
            Fij_var_list.append(self.Fij_var_array[self.nind_iptr[i]:self.nind_iptr[i+1]])

            dc[i] = self.distances[i, self.kstar[i]]
            rr = np.log(self.kstar[i]) - (
                    np.log(prefactor) + self.id_selected * np.log(self.distances[i, self.kstar[i]]))
            Rho[i] = rr
            vj = np.zeros(self.kstar[i])
            for j in range(self.kstar[i]):
                vj[j] = prefactor * (pow(self.distances[i, j + 1], self.id_selected) - pow(
                    self.distances[i, j], self.id_selected))

            vij_list.append(vj)

        l_, Rho = maximise_wPAk(Rho, self.kstar, vij_list, self.dist_indices, Fij_list,
                                Fij_var_list, alpha)

        self.Rho = Rho
        self.Rho -= np.log(self.Nele)

    # ----------------------------------------------------------------------------------------------

    def compute_density_PAk_gCorr_flat(self, alpha=1.,onlyNN=False):
        from duly.utils_.mlmax_pytorch import maximise_wPAk_flatF
        """
        finds the maximum likelihood solution of PAk likelihood + gCorr likelihood with deltaFijs
        computed using the gradients
        """
        # TODO: we need to impement the deltaFijs to be computed as a*l (as in PAk)

        # compute optimal k
        if self.kstar is None:
            self.compute_kstar()

        dc = np.zeros(self.Nele, dtype=float)
        Rho = np.zeros(self.Nele, dtype=float)
        Rho_err = np.zeros(self.Nele, dtype=float)

        prefactor = np.exp(
            self.id_selected / 2. * np.log(np.pi) - gammaln((self.id_selected + 2) / 2))

        vij_list = []

        for i in range(self.Nele):
            dc[i] = self.distances[i, self.kstar[i]]
            rr = np.log(self.kstar[i]) - (
                    np.log(prefactor) + self.id_selected * np.log(self.distances[i, self.kstar[i]]))
            Rho[i] = rr
            vj = np.zeros(self.kstar[i])
            for j in range(self.kstar[i]):
                vj[j] = prefactor * (pow(self.distances[i, j + 1], self.id_selected) - pow(
                    self.distances[i, j], self.id_selected))

            vij_list.append(vj)

            Rho_err[i] = np.sqrt((4 * self.kstar[i] + 2) / (self.kstar[i] * (self.kstar[i] - 1)))

        if self.verb: print("Starting likelihood maximisation")
        sec = time.time()
        l_, Rho = maximise_wPAk_flatF(Rho, Rho_err, self.kstar, vij_list, self.dist_indices, alpha, onlyNN)
        sec2 = time.time()
        if self.verb: print(
            "{0:0.2f} seconds for likelihood maximisation".format(sec2 - sec))


        self.Rho = Rho
        self.Rho -= np.log(self.Nele)

    # ----------------------------------------------------------------------------------------------

    def compute_density_dF_PAk(self):

        # check for deltaFij 
        if self.Fij_array is None: self.compute_deltaFs_grads_semisum()

        if self.verb: 
            print('dF_PAk density estimation started')
            sec=time.time()

        dc = np.zeros(self.Nele, dtype=float)
        Rho = np.zeros(self.Nele, dtype=float)
        Rho_err = np.zeros(self.Nele, dtype=float)
        prefactor = np.exp(
            self.id_selected / 2. * np.log(np.pi) - gammaln((self.id_selected + 2) / 2))

        Rho_min = 9.9E300

        for i in range(self.Nele):
            k = int(self.kstar[i])
            dc[i] = self.distances[i, k]
            Rho[i] = np.log(k) - np.log(prefactor)

            Rho_err[i] = np.sqrt((4 * k + 2) / (k * (k - 1)))
            corrected_rk = 0.
            Fijs = self.Fij_array[self.nind_iptr[i]:self.nind_iptr[i+1]]

            for j in range(1, k ):
                Fij = Fijs[j - 1]
                rjjm1 = self.distances[i, j] ** self.id_selected - self.distances[
                    i, j - 1] ** self.id_selected

                corrected_rk += rjjm1 * np.exp(Fij)  # * (1+Fij)

            Rho[i] -= np.log(corrected_rk)

            if (Rho[i] < Rho_min):
                Rho_min = Rho[i]       

        # Normalise density
        Rho -= np.log(self.Nele)

        self.Rho = Rho
        self.Rho_err = Rho_err
        self.dc = dc

        sec2 = time.time()
        if self.verb: print("{0:0.2f} seconds for dF_PAk density estimation".format(sec2 - sec))

    # ----------------------------------------------------------------------------------------------
    def compute_density_gPAk(self, mode='standard'):
        # compute optimal k
        if self.kstar is None:
            self.compute_kstar()

        dc = np.zeros(self.Nele, dtype=float)
        Rho = np.zeros(self.Nele, dtype=float)
        Rho_err = np.zeros(self.Nele, dtype=float)
        prefactor = np.exp(
            self.id_selected / 2. * np.log(np.pi) - gammaln((self.id_selected + 2) / 2))

        Rho_min = 9.9E300

        self.compute_deltaFs_grad()
        Fij_list = self.Fij_list

        if self.verb: print('gPAk density estimation started')

        if mode == 'standard':

            for i in range(self.Nele):
                k = int(self.kstar[i])
                dc[i] = self.distances[i, k]
                Rho[i] = np.log(k) - np.log(prefactor)

                Rho_err[i] = np.sqrt((4 * self.kstar[i] + 2) / (self.kstar[i] * (self.kstar[i] - 1)))
                corrected_rk = 0.
                Fijs = Fij_list[i]

                for j in range(1, k):
                    Fij = Fijs[j - 1]
                    rjjm1 = self.distances[i, j] ** self.id_selected - self.distances[
                        i, j - 1] ** self.id_selected

                    corrected_rk += rjjm1 * np.exp(Fij)  # * (1+Fij)

                Rho[i] -= np.log(corrected_rk)

                if (Rho[i] < Rho_min):
                    Rho_min = Rho[i]

        elif mode == 'gPAk+':

            vi = np.empty(self.maxk, dtype=float)

            for i in range(self.Nele):
                k = int(self.kstar[i])
                dc[i] = self.distances[i, k]

                rr = np.log(self.kstar[i]) - (
                        np.log(prefactor) + self.id_selected * np.log(self.distances[i, self.kstar[i]]))

                Rho_err[i] = 1. / np.sqrt(k)

                Fijs = Fij_list[i]

                knn = 0
                for j in range(1, k + 1):

                    rjjm1 = self.distances[i, j] ** self.id_selected - self.distances[
                        i, j - 1] ** self.id_selected

                    vi[j - 1] = prefactor * rjjm1

                    if (vi[j - 1] < 1.0E-300):
                        knn = 1

                        break
                if (knn == 0):
                    Rho[i] = MLmax_gPAk(rr, k, vi, Fijs)
                else:
                    Rho[i] = rr

                if (Rho[i] < Rho_min):
                    Rho_min = Rho[i]

        elif mode == 'g+PAk':
            vi = np.empty(self.maxk, dtype=float)

            for i in range(self.Nele):
                k = int(self.kstar[i])
                dc[i] = self.distances[i, k]

                rr = np.log(self.kstar[i]) - (
                        np.log(prefactor) + self.id_selected * np.log(self.distances[i, self.kstar[i]]))

                Rho_err[i] = 1. / np.sqrt(k)

                Fijs = Fij_list[i]

                knn = 0
                for j in range(1, k + 1):

                    rjjm1 = self.distances[i, j] ** self.id_selected - self.distances[
                        i, j - 1] ** self.id_selected

                    vi[j - 1] = prefactor * rjjm1

                    if (vi[j - 1] < 1.0E-300):
                        knn = 1

                        break
                if (knn == 0):
                    Rho[i] = MLmax_gpPAk(rr, k, vi, Fijs)
                else:
                    Rho[i] = rr

                if (Rho[i] < Rho_min):
                    Rho_min = Rho[i]
        else:
            raise ValueError('Please select a valid gPAk mode')

        # Normalise density
        Rho -= np.log(self.Nele)

        self.Rho = Rho
        self.Rho_err = Rho_err
        self.dc = dc

        if self.verb: print('k-NN density estimation finished')

    # ----------------------------------------------------------------------------------------------

    def compute_density_gCorr(self, use_variance=True):
        # TODO: matrix A should be in sparse format!

        # compute changes in free energy
        if self.Fij_array is None: self.compute_deltaFs_grads_semisum()
        
        if self.verb: 
            print('gCorr density estimation started')
            sec=time.time()

        # compute adjacency matrix and cumulative changes
        A = np.zeros((self.Nele, self.Nele))

        deltaFcum_from_i = np.zeros(self.Nele)
        deltaFcum_into_i = np.zeros(self.Nele)

        for i in range(self.Nele):

            Fijs = self.Fij_array[self.nind_iptr[i]:self.nind_iptr[i+1]]
            Fijs_var = self.Fij_var_array[self.nind_iptr[i]:self.nind_iptr[i+1]]

            # deltaFcum_from_i[i] = np.sum(Fijs/Fijs_var)
            deltaFcum_from_i[i] = np.sum(Fijs)
            # TODO: should this be divided by the variance?

            for n_neigh in range(int(self.kstar[i] / 2)):
                j = self.dist_indices[i, n_neigh + 1]

                # update adjacency
                if use_variance:
                    update = 1. / Fijs_var[n_neigh]
                else:
                    update = 1.

                A[i, j] -= update
                A[j, i] -= update

                # deltaFcum_into_i[j] += Fijs[n_neigh]/Fijs_var[n_neigh]
                deltaFcum_into_i[j] += Fijs[n_neigh]
                # TODO: should this be divided by the variance?

        A[np.diag_indices_from(A)] = - np.sum(A, axis=1)

        deltaFcum = deltaFcum_from_i - deltaFcum_into_i

        Rho = np.dot(np.linalg.inv(A), - deltaFcum)

        self.Rho = Rho

        sec2 = time.time()
        if self.verb: print("{0:0.2f} seconds for gCorr density estimation".format(sec2 - sec))

    # ----------------------------------------------------------------------------------------------

    def return_grads(self):
        """[OBSOLETE] Returns the gradient of the log density each point using k* nearest neighbors.
        The gradient is estimated via a linear expansion of the density propagated to the log-density.

        Returns:
            grads (np.ndarray(float)): for each line i contains the gradient components estimated from from point i 

        """
        # compute optimal k
        assert (self.X is not None)

        # compute optimal k
        if self.kstar is None:
            self.compute_kstar()
        d = self.X.shape[1]
        grads = np.zeros((self.X.shape[0], d))

        for i in range(self.X.shape[0]):
            xi = self.X[i]

            mean_vec = np.zeros(d)
            k = int(self.kstar[i])

            for j in range(k):
                xj = self.X[self.dist_indices[i, j + 1]]
                xjmxi = xj - xi
                mean_vec += xjmxi

            mean_vec = mean_vec / k
            r = np.linalg.norm(xjmxi)

            grads[i, :] = (self.id_selected + 2) / r ** 2 * mean_vec

        # grads = gc.compute_grads_from_coords(self.X, self.dist_indices, self.kstar,
        #                                      self.id_selected)

        return grads

    # ----------------------------------------------------------------------------------------------

    def compute_grads(self, comp_covmat=False):
        """Compute the gradient of the log density each point using k* nearest neighbors.
        The gradient is estimated via a linear expansion of the density propagated to the log-density.

        Args:
            k: number of neighbours used to compute the density

        Returns:


        MODIFICARE QUI E ANCHE NEGLI ATTRIBUTI



        """
        # compute optimal k
        if self.kstar is None:
            self.compute_kstar()

        if self.verb:
            print('Estimation of the density gradient started')

        sec = time.time()
        if comp_covmat is False:
            self.grads, self.grads_var = cf.return_grads_and_var_from_coords(self.X, self.dist_indices,
                                                                  self.kstar, self.id_selected)
        else:
            self.grads, self.grads_var = cf.return_grads_and_covmat_from_coords(self.X, self.dist_indices,
                                                                  self.kstar, self.id_selected)
            self.check_grads_covmat = True
        sec2 = time.time()
        if self.verb: print(
            "{0:0.2f} seconds computing gradients".format(sec2 - sec))

    # ----------------------------------------------------------------------------------------------

    def compute_deltaFs_grad(self, extgrads=None, extgrads_covmat=None):
        """Compute deviations deltaFij to standard kNN log-densities at point j as seen from point i using
        a linear expansion (see `compute_grads`).

        Returns:

        """
        # compute optimal k
        if self.kstar is None:
            self.compute_kstar()

        if self.verb:
            print('Estimation of the gradient (linear) corrections deltaFij to the log-density started')

        sec = time.time()
        if self.X is not None:
            if extgrads is None:
                Fij_list, Fij_var_list = cf.return_deltaFs_from_coords(self.X,
                                                                        self.dist_indices,
                                                                        self.kstar,
                                                                        self.id_selected)
            else:
                assert extgrads_covmat is not None
                Fij_list, Fij_var_list = cf.return_deltaFs_from_coords_and_grads(self.X,
                                                                        self.dist_indices,
                                                                        self.kstar,
                                                                        extgrads,
                                                                        extgrads_covmat)

        else:
            print('Warning, falling back to a very slow implementation of the gradient estimation')
            if extgrads is not None:
                print('NOT using the given external gradients')

            Fij_list = []
            Fij_var_list = []

            for i in range(self.Nele):
                k = int(self.kstar[i])

                rk = self.distances[i, k]

                if i % 100 == 0: print(i)

                Fijs = np.empty(k, dtype=float)
                Fijs_var = np.empty(k, dtype=float)

                for j in range(1, k + 1):
                    rij = self.distances[i, j]
                    j_idx = self.dist_indices[i, j]

                    Fij = 0
                    Fij_sq = 0

                    for l in range(1, k + 1):
                        ril = self.distances[i, l]
                        l_idx = self.dist_indices[i, l]

                        idx_jl = np.where(self.dist_indices[j_idx] == l_idx)[0][0]
                        rjl = self.distances[j_idx, idx_jl]
                        # rjl = np.linalg.norm(self.X[j_idx] - self.X[l_idx])
                        Fijl = (rij ** 2 + ril ** 2 - rjl ** 2) / 2.

                        Fij += Fijl
                        Fij_sq += Fijl ** 2

                    Fij = Fij / k
                    Fij_sq = Fij_sq / k

                    Fij = ((self.id_selected + 2) / rk ** 2) * Fij

                    Var_ij = ((self.id_selected + 2) / rk ** 2) ** 2 * Fij_sq - Fij ** 2

                    Fijs[j - 1] = Fij
                    Fijs_var[j - 1] = Var_ij

                Fij_list.append(Fijs)
                Fij_var_list.append(Fijs_var)

        self.Fij_list = Fij_list
        self.Fij_var_list = Fij_var_list

        sec2 = time.time()
        if self.verb: print(
            "{0:0.2f} seconds computing gradient corrections".format(sec2 - sec))

    # ----------------------------------------------------------------------------------------------

    def compute_deltaFs_grads_semisum(self, chi='auto'):
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
        if self.neigh_vector_diffs is None: self.compute_neigh_vector_diffs()

        # check or compute gradients and their covariance matrices
        if self.grads is None: self.compute_grads(comp_covmat=True)

        elif self.check_grads_covmat == False: self.compute_grads(comp_covmat=True)

        if self.verb: print('Estimation of the gradient semisum (linear) corrections deltaFij to the log-density started')
        sec = time.time()

        nspar = self.nind_list.shape[0]

        Fij_array = np.zeros(nspar)
        Fij_var_array = np.zeros(nspar)

        k1 = self.kstar[self.nind_list[:,0]]
        k2 = self.kstar[self.nind_list[:,1]]
        g1 = self.grads[self.nind_list[:,0]]
        g2 = self.grads[self.nind_list[:,1]]
        g_var1 = self.grads_var[self.nind_list[:,0]]
        g_var2 = self.grads_var[self.nind_list[:,1]]
        
        if chi=='auto':
            # check or compute common_neighs
            if self.common_neighs is None: self.compute_common_neighs()

            chi = self.common_neighs/(k1+k2-self.common_neighs)

        else:
            assert (chi >= 0. and chi <=1.), "Invalid value for argument 'chi'"

        Fij_array = 0.5*np.einsum('ij, ij -> i', g1+g2, self.neigh_vector_diffs)
        vari = np.einsum( 'ij, ij -> i', self.neigh_vector_diffs , np.einsum('ijk, ik -> ij', g_var1, self.neigh_vector_diffs))
        varj = np.einsum( 'ij, ij -> i', self.neigh_vector_diffs , np.einsum('ijk, ik -> ij', g_var2, self.neigh_vector_diffs))
        Fij_var_array = 0.25*(vari+varj+2*chi*np.sqrt(vari*varj))

        sec2 = time.time()
        if self.verb: print(
            "{0:0.2f} seconds computing gradient corrections".format(sec2 - sec))

        self.Fij_array = Fij_array
        self.Fij_var_array = Fij_var_array

    # ----------------------------------------------------------------------------------------------


    def return_entropy(self):
        """Compute a very rough estimate of the entropy of the data distribution
        as the average negative log probability.

        Returns:
            H (float): the estimate entropy of the distribution

        """
        assert (self.Rho is not None)

        H = - np.mean(self.Rho)

        return H

# if __name__ == '__main__':
#     X = np.random.uniform(size=(50, 2))
#
#     de = DensityEstimation(coordinates=X)
#
#     de.compute_distances(maxk=25)
#
#     de.compute_id_2NN(decimation=1)
#
#     de.compute_density_kNN(10)
#
#     de.compute_grads()
#
#     print(de.Rho)
