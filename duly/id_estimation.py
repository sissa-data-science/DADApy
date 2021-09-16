import math
import multiprocessing

import numpy as np
import scipy.special as sp
from scipy.optimize import curve_fit

from duly._base import Base
from duly.utils_ import utils as ut

from duly.utils_.utils import compute_nn_distances

cores = multiprocessing.cpu_count()
rng = np.random.default_rng()


class IdEstimation(Base):
    """Estimates the intrinsic dimension of a dataset choosing among various routines.

    Inherits from class Base.

    Attributes:

            intrinsic_dim (int): (rounded) computed intrinsic dimension of the data manifold.

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

        self.intrinsic_dim = None
        self.intrinsic_dim_err = None

    # ----------------------------------------------------------------------------------------------

    def return_id_scaling_r2n(self, range_max=1024, d0=0.001, d1=1000):
        """Compute the id at different scales using the r2n algorithm.

        Args:
                range_max: maximum number neighbourhood considered for the id computations, the largest this number the
                larges is the scale.
                d0: minimum intrinsic dimension considered in the search
                d1: maximum intrinsic dimension considered in the search

        Returns:

        """

        def get_steps(upper_bound, range_max=range_max):
            range_r2 = min(range_max, upper_bound)
            max_step = int(math.log(range_r2, 2))
            return np.array([2 ** i for i in range(max_step)]), range_r2

        if self.distances is not None and range_max < self.maxk:
            steps, _ = get_steps(upper_bound=range_max)
            mus = self.distances[:, steps[1:]] / self.distances[:, steps[:-1]]
            r2s = self.distances[:, np.array([steps[:-1], steps[1:]])]

        elif self.X is not None:
            steps, range_r2 = get_steps(upper_bound=self.N - 1)
            distances, dist_indices, mus, r2s = self._return_mus_scaling(
                range_scaling=range_r2
            )

            # if distances have not been computed save them
            if self.distances is None:
                self.distances = distances
                self.dist_indices = dist_indices
                self.N = distances.shape[0]

        else:
            raise ValueError(
                "You need a coordinate matrix to perform a scaling analysis, or decrease range_max"
            )

        # array of ids (as a function of the average distance to a point)
        ids_scaling = np.zeros(mus.shape[1])
        # array of error estimates (via fisher information)
        ids_scaling_err = np.zeros(mus.shape[1])
        # array of average 'first' and 'second' neighbor distances, relative to each id estimate
        r2s_scaling = np.mean(r2s, axis=(0, 1))

        # compute IDs via maximum likelihood (and their error) for all the scales up to range_scaling
        for i in range(mus.shape[1]):
            n1 = 2 ** i

            ids_scaling[i], ids_scaling_err[i] = self.compute_id_r2n(
                n1, d0=d0, d1=d1, mus=mus[:, i], return_id=True
            )

        return ids_scaling, ids_scaling_err, r2s_scaling

    # ----------------------------------------------------------------------------------------------

    def compute_id_r2n(self, n1, d0=0.001, d1=1000, mus=None, return_id=False):
        assert n1 * 2 < self.maxk

        if mus is None:
            mus = np.log(self.distances[:, n1 * 2] / self.distances[:, n1])

        id = ut._argmax_loglik(self.dtype, d0, d1, mus, n1, 2 * n1, self.N, eps=1.0e-7)
        id_err = (1 / ut._fisher_info_scaling(id, mus, n1, 2 * n1)) ** 0.5

        self.intrinsic_dim = id
        self.intrinsic_dim_err = id_err

        if return_id:
            return id, id_err

    def compute_id_2NN(
        self, algorithm="base", fraction=0.9, N_subset=None, return_id=False
    ):
        """Compute intrinsic dimension using the 2NN algorithm

        Args:
                algorithm: 'base' to perform the linear fit, 'ml' to perform maximum likelihood
                fraction: fraction of mus that will be considered for the estimate (discard highest mus)
                N_subset: Can be used to change the scale at which one is looking at the data

        Returns:

        """
        if N_subset is None:
            N_subset = self.N

        nrep = int(np.round(self.N / N_subset))
        ids = np.zeros(nrep)
        r2s = np.zeros((nrep, 2))

        def _compute_id_2NN_single(mus, fraction):

            N = mus.shape[0]
            N_eff = int(N * fraction)

            mus_reduced = np.sort(mus)[:N_eff]

            if algorithm == "ml":
                intrinsic_dim = N / np.sum(mus)

            elif algorithm == "base":
                y = -np.log(1 - np.arange(1, N_eff + 1) / N)

                def func(x, m):
                    return m * x

                intrinsic_dim, _ = curve_fit(func, mus_reduced, y)

            else:
                raise ValueError("Please select a valid algorithm type")

            return intrinsic_dim

        for i in range(nrep):
            idx = np.random.choice(self.N, size=N_subset, replace=False)

            d, _ = compute_nn_distances(
                self.X[idx], min(N_subset, self.maxk), self.metric, self.p, self.period
            )

            mus = np.log(d[:, 2] / d[:, 1])
            id = _compute_id_2NN_single(mus, fraction)

            r2s[i] = np.mean(d[:, np.array([1, 2])])
            ids[i] = id

        id = np.mean(ids)
        stderr = np.std(ids) / len(ids) ** 0.5

        if self.verb:
            print(f"ID estimation finished: selecting ID of {id} +- {stderr}")

        self.intrinsic_dim = id
        self.intrinsic_dim_err = stderr

        if return_id:
            return (
                id,
                stderr,
                np.mean(r2s),
            )

    def return_id_scaling_2NN(
        self,
        N_min=10,
        algorithm="base",
        fraction=0.9,
    ):
        max_ndec = int(math.log(self.N, 2)) - 1
        Nsubsets = np.round(self.N / np.array([2 ** i for i in range(max_ndec)]))
        Nsubsets = Nsubsets.astype(int)

        if N_min is not None:
            Nsubsets = Nsubsets[Nsubsets > N_min]

        ids_scaling = np.zeros(Nsubsets.shape[0])
        ids_scaling_err = np.zeros(Nsubsets.shape[0])
        r2s_scaling = np.zeros((Nsubsets.shape[0]))

        for i, N_subset in enumerate(Nsubsets):
            ids_scaling[i], ids_scaling_err[i], r2s_scaling[i] = self.compute_id_2NN(
                algorithm=algorithm,
                fraction=fraction,
                N_subset=N_subset,
                return_id=True,
            )

        return ids_scaling, ids_scaling_err, r2s_scaling

    # ----------------------------------------------------------------------------------------------

    def compute_id_gammaprior(self, alpha=2, beta=5):
        if self.distances is None:
            self.compute_distances()

        if self.verb:
            print(
                "ID estimation started, using alpha = {} and beta = {}".format(
                    alpha, alpha
                )
            )

        distances_used = self.distances

        sum_log_mus = np.sum(np.log(distances_used[:, 2] / distances_used[:, 1]))

        alpha_post = alpha + self.N
        beta_post = beta + sum_log_mus

        mean_post = alpha_post / beta_post
        std_post = np.sqrt(alpha_post / beta_post ** 2)
        mode_post = (alpha_post - 1) / beta_post

        self.id_alpha_post = alpha_post
        self.id_beta_post = beta_post
        self.id_estimated_mp = mean_post
        self.id_estimated_mp_std = std_post
        self.id_estimated_map = mode_post
        self.intrinsic_dim = int(np.around(self.id_estimated_mp, decimals=0))

    # ----------------------------------------------------------------------------------------------

    def fix_rk(self, rk=None, ratio=None):
        """Computes the k points within the given rk and n points within given rn.

        For each point, computes the number self.k of points within a sphere of radius rk
        and the number self.n within an inner sphere of radius rn=rk*ratio. It also provides
        a mask to take into account those points for which the statistics might be wrong, i.e.
        k == self.maxk, meaning that all available points were selected. If self.maxk is equal
        to the number of points of the dataset no mask will be applied

        Args:
                rk (float or int): external shell radius
                ratio (float,optional): ratio between internal and external radii of the shells

        Returns:

        """
        # checks-in and initialisations
        if self.distances is None:
            self.compute_distances()

        if ratio is not None:
            self.set_r(ratio)
        else:
            assert (
                self.r is not None
            ), "set self.r or insert a value for the ratio parameter"

        if rk is not None:
            self.set_rk(rk)
        else:
            assert (
                self.rk is not None
            ), "set self.R or insert a value for the rk parameter"

        # routine
        self.rn = self.rk * self.r
        self.k = (self.distances <= self.rk).sum(axis=1)
        self.n = (self.distances <= self.rn).sum(axis=1)

        # checks-out
        if self.maxk < self.N:
            # if not all possible NN were taken into account (i.e. maxk < N) and k is equal to self.maxk
            # or distances[:,-1]<rk, it is likely that there are other points within rk that are not being
            # considered and thus potentially altering the statistics -> neglect them through self.mask
            # in the calculation of likelihood
            self.mask = self.distances[:, -1] > self.rk  # or self.k == self.maxk

            if np.any(~self.mask):
                print(
                    "NB: for "
                    + str(sum(~(self.mask)))
                    + " points, the counting of k could be wrong, "
                    + "as more points might be present within the selected rk. In order not to affect "
                    + "the statistics a mask is provided to remove them from the calculation of the "
                    + "likelihood or posterior.\nConsider recomputing NN with higher maxk or lowering rk."
                )

        else:
            self.mask = np.ones(self.N, dtype=bool)

    # ----------------------------------------------------------------------------------------------

    def compute_id_binomial_rk(
        self, rk=None, ratio=None, subset=None, method="bayes", plot=False
    ):
        """Calculate Id using the binomial estimator by fixing the eternal radius for all the points

        In the estimation of d one has to remove the central point from the counting of n and k
        as it generates the process but it is not effectively part of it

        Args:
                rk (float): radius of the external shell
                ratio (float): ratio between internal and external shell
                subset (int): choose a random subset of the dataset to make the Id estimate
                method (str, default='bayes'): choose method between 'bayes' and 'mle'. The bayesian estimate
                                            gives the mean value and std of d, while mle only the max of the likelihood
                plot (bool, default=False): if True plots the posterior and initialise self.posterior_domain and self.posterior

        """
        # checks-in and initialisations
        if ratio is not None:
            self.set_r(ratio)
        else:
            assert (
                self.r is not None
            ), "set self.r or insert a valure for the ratio parameter"

        if rk is not None:
            self.set_rk(rk)
        else:
            assert (
                self.rk is not None
            ), "set self.R or insert a valure for the rk parameter"

        self.rn = self.rk * self.r

        # routine
        self.fix_rk()

        n_eff = self.n[self.mask]
        k_eff = self.k[self.mask]

        if subset is not None:
            assert isinstance(
                subset, (np.integer, int)
            ), "subset needs to be an integer"
            if subset < len(n_eff):
                subset = rng.choice(len(n_eff), subset, replace=False, shuffle=False)
                n_eff = n_eff[subset]
                k_eff = k_eff[subset]

        E_n = n_eff.mean()
        if E_n == 1.0:
            print(
                "no points in the inner shell, returning 0. Consider increasing rk and/or the ratio"
            )
            self.id_estimated_binom = 0
            return 0

        if method == "mle":
            self.id_estimated_binom = np.log((E_n - 1) / (k_eff.mean() - 1)) / np.log(
                self.r
            )
        elif method == "bayes":
            (
                self.id_estimated_binom,
                self.id_estimated_binom_std,
                self.posterior_domain,
                self.posterior,
            ) = _beta_prior(k_eff - 1, n_eff - 1, self.r, plot=plot)
        else:
            print("select a proper method for id computation")
            return 0

    # ----------------------------------------------------------------------------------------------

    def fix_k(self, k_eff=None, ratio=None):
        """Computes rk, rn, n for each point given a selected value of k

        This routine computes the external radius rk, internal radius rn and internal points n
        given a value k (the number of NN to consider).

        Args:
                k_eff (int, default=self.maxk): selected number of NN
                ratio (float, ): ratio among rn and rk

        Returns:

        """
        # checks-in and initialisations
        if self.distances is None:
            self.compute_distances()

        if ratio is not None:
            self.set_r(ratio)
        else:
            assert (
                self.r is not None
            ), "set self.r or insert a value for the parameter ratio"

        if k_eff is not None:
            assert (
                k_eff < self.maxk
            ), "You first need to recompute the distances with the proper amount on NN"
        else:
            k_eff = self.maxk - 1

        # routine
        self.k = k_eff
        self.rk = self.distances[:, self.k]  # k NN -> k-1 position in the array
        self.rn = self.rk * self.r
        self.n = (self.distances <= self.rn.reshape(self.N, 1)).sum(axis=1)

        self.mask = np.ones(self.N, dtype=bool)

    # --------------------------------------------------------------------------------------

    def compute_id_binomial_k(
        self, k=None, ratio=None, subset=None, method="bayes", plot=False
    ):
        """Calculate Id using the binomial estimator by fixing the number of neighbours

        As in the case in which one fixes rk, also in this version of the estimation
        one removes the central point from n and k. Furthermore one has to remove also
        the k-th NN, as it plays the role of the distance at which rk is taken.
        So if k=5 it means the 5th NN from the central point will be considered,
        taking into account 6 points though (the central one too). This means that
        in principle k_eff = 6, to which I'm supposed to subtract 2. For this reason
        in the computation of the MLE we have directly k-1, which explicitely would be k_eff-2

        Args:
                k (int): order of neighbour that set the external shell
                ratio (float): ratio between internal and external shell
                subset (int): choose a random subset of the dataset to make the Id estimate
                method (str, default='bayes'): choose method between 'bayes' and 'mle'. The bayesian estimate
                                            gives the mean value and std of d, while mle only the max of the likelihood
                plot (bool, default=False): if True plots the posterior and initialise self.posterior_domain and self.posterior
        """
        # checks-in and initialisations
        if ratio is not None:
            self.set_r(ratio)
        else:
            assert (
                self.r is not None
            ), "set self.r or insert a value for the ratio parameter"

        if k is not None:
            assert (
                k < self.maxk
            ), "You first need to recompute the distances with the proper number of NN"
        else:
            k = self.maxk - 1

        # routine
        self.fix_k(k)
        n_eff = self.n

        if subset is not None:
            assert isinstance(
                subset, (np.integer, int)
            ), "subset needs to be an integer"
            if subset < len(self.n):
                subset = rng.choice(len(self.n), subset, replace=False, shuffle=False)
                n_eff = self.n[subset]

        E_n = n_eff.mean()
        if E_n == 1.0:
            print(
                "no points in the inner shell, returning 0\n. Consider increasing rk and/or the ratio"
            )
            self.id_estimated_binom = 0
            return 0

        if method == "mle":
            self.id_estimated_binom = np.log((E_n - 1) / (k - 1)) / np.log(self.r)
        elif method == "bayes":
            (
                self.id_estimated_binom,
                self.id_estimated_binom_std,
                self.posterior_domain,
                self.posterior,
            ) = _beta_prior(k - 1, n_eff - 1, self.r, plot=plot)
        else:
            print("select a proper method for id computation")
            return 0

    # ----------------------------------------------------------------------------------------------

    # theoretical lower bound limit for the variance on the continuous estimator
    def CramerRao(self, d=None, k=None, r=None, N=None):

        if d is None:
            assert (
                self.id_estimated_binom is not None
            ), "You have to compute the id using the binomial estimator first!"
            d = self.intrinsic_dim_binom
        if k is None:
            k = self.maxk
        if r is None:
            r = self.r
        if N is None:
            N = self.N

        self.VarCramerRao = (r ** (-d) - 1) / (k * N * np.log(r) ** 2)

    # ----------------------------------------------------------------------------------------------
    def set_id(self, d):
        assert d > 0, "cannot support negative dimensions (yet)"
        self.intrinsic_dim = d

    # ----------------------------------------------------------------------------------------------
    def set_r(self, r):
        assert 0 < r < 1, "select a proper ratio, 0<r<1"
        self.r = r

    # ----------------------------------------------------------------------------------------------
    def set_rk(self, R):
        assert 0 < R, "select a proper rk>0"
        self.rk = R


# ----------------------------------------------------------------------------------------------


def _beta_prior(k, n, r, a0=1, b0=1, plot=False):
    """Compute the posterior distribution of d given the input aggregates
    Since the likelihood is given by a binomial distribution, its conjugate prior is a beta distribution.
    However, the binomial is defined on the ratio of volumes and so do the beta distribution. As a
    consequence one has to change variable to have the distribution over d

    Args:
            k (nd.array(int) or int): number of points within the external shells
            n (nd.array(int)): number of points within the internal shells
            r (float): ratio between shells' radii
            a0 (float): beta distribution parameter, default =1 for flat prior
            b0 (float): prior initialiser, default =1 for flat prior
            plot (bool,default=False): plot the posterior and give a numerical estimate other than the analytical one
    Returns:
            mean_bayes (float): mean value of the posterior
            std_bayes (float): std of the posterior
    """
    from scipy.special import beta as beta_f
    from scipy.stats import beta as beta_d

    D_MAX = 300.0
    D_MIN = 0.0001

    a = a0 + n.sum()
    if isinstance(k, (np.int, int)):
        b = b0 + k * n.shape[0] - n.sum()
    else:
        b = b0 + k.sum() - n.sum()
    posterior = beta_d(a, b)

    if plot:
        import matplotlib.pyplot as plt

        def p_d(d):
            return abs(posterior.pdf(r ** d) * (r ** d) * np.log(r))

        dx = 0.1
        d_left = D_MIN
        d_right = D_MAX + dx + d_left
        d_range = np.arange(d_left, d_right, dx)
        P = np.array([p_d(di) for di in d_range]) * dx
        mask = P != 0
        elements = mask.sum()
        counter = 0
        # if less than 3 points !=0 are found, reduce the interval
        while elements < 3:
            dx /= 10
            d_range = np.arange(d_left, d_right, dx)
            P = np.array([p_d(di) for di in d_range]) * dx
            mask = P != 0
            elements = mask.sum()
            counter += 1

        # with more than 3 points !=0 we can restrict the domain and have a smooth distribution
        # I choose 1000 points but such quantity can be varied according to necessity
        ind = np.where(mask)[0]
        d_left = d_range[ind[0]] - 0.5 * dx if d_range[ind[0]] - dx > 0 else D_MIN
        d_right = d_range[ind[-1]] + 0.5 * dx
        d_range = np.linspace(d_left, d_right, 1000)
        dx = (d_right - d_left) / 1000
        P = np.array([p_d(di) for di in d_range]) * dx

        plt.plot(d_range, P)
        plt.xlabel("d")
        plt.ylabel("P(d)")
        E_d_emp = (d_range * P).sum()
        S_d_emp = np.sqrt((d_range * d_range * P).sum() - E_d_emp * E_d_emp)
        print("empirical average:\t", E_d_emp, "\nempirical std:\t\t", S_d_emp)

    E_d = (sp.digamma(a) - sp.digamma(a + b)) / np.log(r)
    S_d = np.sqrt((sp.polygamma(1, a) - sp.polygamma(1, a + b)) / np.log(r) ** 2)

    if plot:
        return E_d, S_d, d_range, P
    else:
        return E_d, S_d, None, None
