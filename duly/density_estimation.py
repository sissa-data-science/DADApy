import time

from scipy.special import gammaln

from duly.id_estimation import *


class DensityEstimation(IdEstimation):

    def __init__(self, coordinates=None, distances=None, maxk=None, verbose=False, njobs=cores):
        super().__init__(coordinates=coordinates, distances=distances, maxk=maxk, verbose=verbose,
                         njobs=njobs)

    def compute_density_kNN(self, k=3):
        """Compute the density of of each point using a simple kNN estimator

        Args:
            k: number of neighbours used to compute the density

        Returns:

        """
        assert (self.id_selected is not None)

        if self.verb: print('k-NN density estimation started (k={})'.format(k))

        kstar = np.empty(self.Nele, dtype=int)
        dc = np.empty(self.Nele, dtype=float)
        Rho = np.empty(self.Nele, dtype=float)
        Rho_err = np.empty(self.Nele, dtype=float)
        prefactor = np.exp(
            self.id_selected / 2. * np.log(np.pi) - gammaln((self.id_selected + 2) / 2))
        Rho_min = 9.9E300

        for i in range(self.Nele):
            kstar[i] = k
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

    def compute_kstar(self, Dthr=23.92):
        """Computes the density of each point using a simple kNN estimator with an optimal choice of k.

        Args:
            Dthr: Likelihood ratio parameter used to compute optimal k, the value of Dthr=23.92 corresponds
            to a p-value of 1e-6.

        Returns:

        """
        if self.id_selected is None: self.compute_id()

        if self.verb: print('kstar estimation started, Dthr = {}'.format(Dthr))

        # Dthr = 23.92812698  # this threshold value corresponds to being sure within a p-value
        # of 1E-6 that the k-NN densities, do not touch unless you really know what you are doing

        # Array inizialization for kstar
        kstar = np.empty(self.Nele, dtype=int)
        prefactor = np.exp(
            self.id_selected / 2. * np.log(np.pi) - gammaln((self.id_selected + 2) / 2))

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
        if self.verb: print(
            "{0:0.2f} seconds finding the optimal k for all the points".format(sec2 - sec))

        self.kstar = kstar

    def compute_density_kstarNN(self):
        if self.kstar is None: self.compute_kstar()
        kstar = self.kstar

        if self.verb: print('kstar-NN density estimation started')

        dc = np.empty(self.Nele, dtype=float)
        Rho = np.empty(self.Nele, dtype=float)
        Rho_err = np.empty(self.Nele, dtype=float)
        prefactor = np.exp(
            self.id_selected / 2. * np.log(np.pi) - gammaln((self.id_selected + 2) / 2))

        Rho_min = 9.9E300

        for i in range(self.Nele):
            k = kstar[i]
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

    def compute_density_PAk(self, method='NR'):
        from cython_ import cython_functions as cf

        # options for method:
        #   - "NR"=Newton-Raphson implemented in cython
        #   - "NM"=Nelder-Mead scipy built-in
        #   - "For"=Newton-Raphson implemented in Fortran

        # compute optimal k
        if self.kstar is None: self.compute_kstar()
        kstar = self.kstar

        if self.verb: print('PAk density estimation started')

        dc = np.empty(self.Nele, dtype=float)
        Rho = np.empty(self.Nele, dtype=float)
        Rho_err = np.empty(self.Nele, dtype=float)
        vi = np.empty(self.maxk, dtype=float)
        prefactor = np.exp(
            self.id_selected / 2. * np.log(np.pi) - gammaln((self.id_selected + 2) / 2))
        Rho_min = 9.9E300

        sec = time.time()
        for i in range(self.Nele):
            dc[i] = self.distances[i, kstar[i]]
            rr = np.log(kstar[i]) - (
                    np.log(prefactor) + self.id_selected * np.log(self.distances[i, kstar[i]]))
            knn = 0
            for j in range(kstar[i]):
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
                    Rho[i] = cf._nrmaxl(rr, kstar[i], vi, self.maxk)
                elif (method == 'NM'):
                    from mlmax import MLmax
                    Rho[i] = MLmax(rr, kstar[i], vi)
                else:
                    raise ValueError("Please choose a valid method")
                # Rho[i] = NR.nrmaxl(rr, kstar[i], vi, self.maxk) # OLD FORTRAN
            else:
                Rho[i] = rr
            if (Rho[i] < Rho_min):
                Rho_min = Rho[i]

            Rho_err[i] = np.sqrt((4 * kstar[i] + 2) / (kstar[i] * (kstar[i] - 1)))

        sec2 = time.time()
        if self.verb: print(
            "{0:0.2f} seconds optimizing the likelihood for all the points".format(sec2 - sec))

        # Normalise density
        Rho -= np.log(self.Nele)

        self.Rho = Rho
        self.Rho_err = Rho_err
        self.dc = dc
        self.kstar = kstar

        if self.verb: print('PAk density estimation finished')

    def compute_density_kstarNN_gCorr(self, alpha=1., gauss_approx=False, Fij_type='grad'):
        """
        finds the minimum of the
        """
        from mlmax_pytorch import maximise
        # Fij_types: 'grad', 'zero', 'PAk'
        # TODO: we need to implement a gCorr term with the deltaFijs equal to zero

        # compute optimal k
        if self.kstar is None: self.compute_kstar()
        kstar = self.kstar

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

            dc = np.array([self.distances[i, kstar[i]] for i in range(self.Nele)])

            Vis = prefactor * (dc ** self.id_selected)

            # get good initial conditions for the optimisation
            Fis = np.array([np.log(kstar[i]) - np.log(Vis[i]) for i in range(self.Nele)])

            # optimise the likelihood using pytorch
            l_, Rho = maximise(Fis, kstar, Vis, self.dist_indices, Fij_list, Fij_var_list, alpha)

        # normalise density
        Rho -= np.log(self.Nele)

        self.Rho = Rho

    def compute_density_PAk_gCorr(self, alpha=1.):
        from mlmax_pytorch import maximise_wPAk
        """
        finds the maximum likelihood solution of PAk likelihood + gCorr likelihood with deltaFijs
        computed using the gradients
        """
        # TODO: we need to impement the deltaFijs to be computed as a*l (as in PAk)

        # compute optimal k
        if self.kstar is None: self.compute_kstar()
        kstar = self.kstar

        # compute changes in free energy
        if self.Fij_list is None: self.compute_deltaFs_grad()
        Fij_list = self.Fij_list
        Fij_var_list = self.Fij_var_list

        dc = np.empty(self.Nele, dtype=float)
        Rho = np.empty(self.Nele, dtype=float)

        prefactor = np.exp(
            self.id_selected / 2. * np.log(np.pi) - gammaln((self.id_selected + 2) / 2))

        vij_list = []

        for i in range(self.Nele):
            dc[i] = self.distances[i, kstar[i]]
            rr = np.log(kstar[i]) - (
                    np.log(prefactor) + self.id_selected * np.log(self.distances[i, kstar[i]]))
            Rho[i] = rr
            vj = np.zeros(kstar[i])
            for j in range(kstar[i]):
                vj[j] = prefactor * (pow(self.distances[i, j + 1], self.id_selected) - pow(
                    self.distances[i, j], self.id_selected))

            vij_list.append(vj)

        l_, Rho = maximise_wPAk(Rho, kstar, vij_list, self.dist_indices, Fij_list,
                                Fij_var_list, alpha)

        self.Rho = Rho
        self.Rho -= np.log(self.Nele)

    def compute_density_gPAk(self, mode='standard'):
        from mlmax import MLmax_gPAk, MLmax_gpPAk
        # compute optimal k
        if self.kstar is None: self.compute_kstar()
        kstar = self.kstar

        dc = np.empty(self.Nele, dtype=float)
        Rho = np.empty(self.Nele, dtype=float)
        Rho_err = np.empty(self.Nele, dtype=float)
        prefactor = np.exp(
            self.id_selected / 2. * np.log(np.pi) - gammaln((self.id_selected + 2) / 2))

        Rho_min = 9.9E300

        self.compute_deltaFs_grad()
        Fij_list = self.Fij_list

        if self.verb: print('gPAk density estimation started')

        if mode == 'standard':

            for i in range(self.Nele):
                k = int(kstar[i])
                dc[i] = self.distances[i, k]
                Rho[i] = np.log(k) - np.log(prefactor)

                Rho_err[i] = 1. / np.sqrt(k)
                corrected_rk = 0.
                Fijs = Fij_list[i]

                for j in range(1, k + 1):
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
                k = int(kstar[i])
                dc[i] = self.distances[i, k]

                rr = np.log(kstar[i]) - (
                        np.log(prefactor) + self.id_selected * np.log(self.distances[i, kstar[i]]))

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
                k = int(kstar[i])
                dc[i] = self.distances[i, k]

                rr = np.log(kstar[i]) - (
                        np.log(prefactor) + self.id_selected * np.log(self.distances[i, kstar[i]]))

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

    def compute_density_gCorr(self, use_variance=True):
        # TODO: matrix A should be in sparse format!

        # compute optimal k
        if self.kstar is None: self.compute_kstar()

        # compute changes in free energy
        if self.Fij_list is None: self.compute_deltaFs_grad()
        Fij_list = self.Fij_list
        Fij_var_list = self.Fij_var_list

        # compute adjacency matrix and cumulative changes
        A = np.zeros((self.Nele, self.Nele))

        deltaFcum_from_i = np.zeros(self.Nele)
        deltaFcum_into_i = np.zeros(self.Nele)

        for i in range(self.Nele):

            Fijs = Fij_list[i]
            Fijs_var = Fij_var_list[i] + 1.e-4

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

    def return_grads(self):
        assert (self.X is not None)

        # compute optimal k
        if self.kstar is None: self.compute_kstar()
        d = self.X.shape[1]
        grads = np.zeros((self.X.shape[0], d))

        for i in range(self.X.shape[0]):
            xi = self.X[i]

            mean_vec = np.zeros(d)
            kstar = int(self.kstar[i])

            for j in range(kstar):
                xj = self.X[self.dist_indices[i, j + 1]]
                xjmxi = xj - xi
                mean_vec += xjmxi

            mean_vec = mean_vec / kstar
            r = np.linalg.norm(xjmxi)

            grads[i, :] = (self.id_selected + 2) / r ** 2 * mean_vec

        # grads = gc.compute_grads_from_coords(self.X, self.dist_indices, self.kstar,
        #                                      self.id_selected)

        return grads

    def compute_deltaFs_grad(self):
        from cython_ import cython_functions as cf
        # compute optimal k
        if self.kstar is None: self.compute_kstar()

        if self.verb: print('Estimation of the density gradient started')

        sec = time.time()
        if self.X is not None:
            Fij_list, Fij_var_list = cf.compute_deltaFs_from_coords(self.X, self.dist_indices,
                                                                    self.kstar, self.id_selected)

        else:
            print('Warning, falling back to a very slow implementation of the gradient estimation')

            Fij_list = []
            Fij_var_list = []

            kstar = self.kstar
            for i in range(self.Nele):
                k = int(kstar[i])

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

    def return_entropy(self):
        assert (self.Rho is not None)

        H = - np.mean(self.Rho)

        return H


if __name__ == '__main__':
    X = np.random.uniform(size=(50, 2))

    de = DensityEstimation(coordinates=X)

    de.compute_distances(maxk=25)

    de.compute_id()

    de.compute_density_kNN(10)

    print(de.Rho)
