import multiprocessing

import numpy as np
import scipy.special as sp
from scipy.optimize import minimize_scalar as SMin

from duly._base import Base
from duly.utils_ import discrete_functions as df
from duly.utils_ import utils as ut

cores = multiprocessing.cpu_count()
rng = np.random.default_rng()


class IdDiscrete(Base):
    """Estimates the intrinsic dimension of a dataset choosing among various routines.

    Inherits from class Base.

    Attributes:

            id_selected (int): (rounded) selected intrinsic dimension after each estimate. Parameter used for density estimation
            id_estimated_D (float):id estimated using Diego's routine
            id_estimated_2NN (float): 2NN id estimate

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

        if weights is not None:
            self.set_w(weights)
            self.is_w = True
        else:
            self.is_w = False

        self.Lk = None
        self.Ln = None

    # ----------------------------------------------------------------------------------------------

    def fix_Lk(self, Lk=None, Ln=None):
        """Computes the k points within the given Rk and n points within given Rn.

        For each point, computes the number self.k of points within a sphere of radius Rk
        and the number self.n within an inner sphere of radius Rk. It also provides
        a mask to take into account those points for which the statistics might be wrong, i.e.
        k == self.maxk, meaning that all available points were selected or when we are not
        sure to have all the point of the selected shell. If self.maxk is equal
        to the number of points of the dataset no mask will be applied, as all the point
        were surely taken into account.
        Eventually add the weights of the points if they have an explicit multeplicity

        Args:
                Lk (int): external shell radius
                Ln (int): internal shell radius
                w (ndarray(int), optional): array of multeplicities of points

        Returns:

        """
        # checks-in and intialisations
        assert (
            self.distances is not None
        ), "first compute distances with the proper metric (manhattan of hamming presumably)"

        if Lk is not None and Ln is not None:
            self.set_Lk_Ln(Lk, Ln)
        else:
            assert (
                self.Lk is not None and self.Ln is not None
            ), "set self.Lk and self.Ln or insert proper values for the Lk and Ln parameters"

        # compute k and n
        if self.is_w is False:
            self.k = (self.distances <= self.Lk).sum(axis=1)
            self.n = (self.distances <= self.Ln).sum(axis=1)
        else:
            assert (
                self.weights is not None
            ), "first insert the weights if you want to use them!"
            self.k = np.array(
                [
                    sum(self.weights[self.dist_indices[i][el]])
                    for i, el in enumerate(self.distances <= self.Lk)
                ],
                dtype=np.int,
            )
            self.n = np.array(
                [
                    sum(self.weights[self.dist_indices[i][el]])
                    for i, el in enumerate(self.distances <= self.Ln)
                ],
                dtype=np.int,
            )

        # checks-out
        if self.maxk < self.Nele:
            # if not all possible NN were taken into account (i.e. maxk < Nele) and k is equal to self.maxk
            # or distances[:,-1]<Lk, it is likely that there are other points within Lk that are not being
            # considered and thus potentially altering the statistics -> neglect them through self.mask
            # in the calculation of likelihood
            self.mask = self.distances[:, -1] > self.Lk  # or self.k == self.maxk

            if np.any(~self.mask):
                print(
                    "NB: for "
                    + str(sum(~(self.mask)))
                    + " points, the counting of k could be wrong, "
                    + "as more points might be present within the selected Rk. In order not to affect "
                    + "the statistics a mask is provided to remove them from the calculation of the "
                    + "likelihood or posterior.\nConsider recomputing NN with higher maxk or lowering Rk."
                )

        else:
            self.mask = np.ones(self.Nele, dtype=bool)

    # ----------------------------------------------------------------------------------------------

    def compute_id_binomial_Lk(self, Lk=None, Ln=None, method="bayes", plot=True):
        """Calculate Id using the binomial estimator by fixing the eternal radius for all the points

        In the estimation of d one has to remove the central point from the counting of n and k
        as it generates the process but it is not effectively part of it

        Args:
                Rk (float): radius of the external shell
                ratio (float): ratio between internal and external shell
                method (str, default 'bayes'): choose method between 'bayes' and 'mle'. The bayesian estimate
                                                                           gives the mean value and std of d, while mle only the max of the likelihood

        """
        # checks-in and initialisations
        assert (
            self.distances is not None
        ), "first compute distances with the proper metric (manhattan of hamming presumably)"

        if Lk is not None and Ln is not None:
            self.set_Lk_Ln(Lk, Ln)
        else:
            assert (
                self.Lk is not None and self.Ln is not None
            ), "set self.Lk and self.Ln or insert proper values for the Lk and Ln parameters"

        # routine
        self.fix_Lk()

        n_eff = self.n[self.mask]
        k_eff = self.k[self.mask]

        if self.verb:
            print("n and k computed")

        E_n = n_eff.mean()
        if E_n == 1.0:
            print(
                "no points in the inner shell, returning 0. Consider increasing Rk and/or the ratio"
            )
            return 0

        if method == "mle":
            if self.is_w:
                ww = self.weights[self.mask]
            else:
                ww = 1
            self.id_estimated_binom = SMin(    df.compute_binomial_logL,
        args=(self.Lk,k_eff-1,self.Ln,n_eff-1,True,ww),
        bounds=(0.0001,100.),method='bounded'    ).x

        elif method == "bayes":
            if self.is_w:
                k_tot = self.weights[self.mask] * (k_eff - 1)
                n_tot = self.weights[self.mask] * (n_eff - 1)
            else:
                k_tot = k_eff - 1
                n_tot = n_eff - 1

            if self.verb:
                print("startin bayesian estimation")

            (
                self.id_estimated_binom,
                self.id_estimated_binom_std,
                self.posterior_domain,
                self.posterior,
            ) = _beta_prior_d(
                k_tot, n_tot, self.Lk, self.Ln, plot=plot, verbose=self.verb
            )
        else:
            print("select a proper method for id computation")
            return 0

    # ----------------------------------------------------------------------------------------------

    def fix_k(self, k_eff=None, ratio=0.7):
        """Computes Rk, Rn, n for each point given a selected value of k

        This routine computes external radii Lk, internal radii Ln and internal points n.
        It also ensure that Lk is scaled onto the most external shell
        for which we are sure to have all points.
        NB As a cosequence the k will be point dependent

        Args:
                k (int, default=self.k_max): selected (max) number of NN
                ratio (float, efault = 0.5): approximate ratio among Ln and Lk

        Returns:
        """

        # TODO: what if we have multiplicity???? use it straightly on the plain points
        # 		or take it into account when counting the number of NN?

        # checks-in and initialisations
        assert (
            self.distances is not None
        ), "first compute distances with the proper metric (manhattan of hamming presumably)"

        assert ratio > 0 and ratio < 1, "set a proper value for the ratio"

        if k_eff is not None:
            assert (
                k_eff <= self.maxk
            ), "A k_eff > maxk was selected,\
			 	recompute the distances with the proper amount on NN to see points up to that k_eff"
        else:
            k_eff = self.maxk

        # routine
        self.Lk, self.k, self.Ln, self.n = (
            np.zeros(self.Nele),
            np.zeros(self.Nele, dtype=np.int64),
            np.zeros(self.Nele),
            np.zeros(self.Nele, dtype=np.int64),
        )
        self.mask = np.ones(self.Nele, dtype=bool)

        # cut distances at the k-th NN and cycle over each point
        for i, dist_i in enumerate(self.distances[:, :k_eff]):

            # case 1: all possible neighbours have been considered, no extra possible unseen points
            if k_eff == self.Nele:
                Lk_temp = dist_i[-1]
                k_temp = k_eff
            # case 2: only some NN are considered -> work on surely "complete" shells
            else:
                # Group distances according to shell. Returns radii of shell and cumulative number of points at that radius.
                # The index at which a distance is first met == number of points at smaller distance
                # EXAMPLE:
                # a = np.array([0,1,1,2,3,3,4,4,4,6,6]) 	and suppose it would go on as 6,6,6,7,8...
                # un, ind = np.unique(a, return_index=True)
                # un: array([0, 1, 2, 3, 4, 6])
                # ind: array([0, 1, 3, 4, 6, 9])
                unique, index = np.unique(dist_i, return_index=True)
                # Lk=0 may happen when we have smtg like dist_i = [0,0,0,0,0] or dist_i = [0,3,3,3,3,3]. As we don't have the full
                # information for at least two shells, we skip the point
                if unique.shape[0] < 3:
                    self.mask[i] = False
                    continue

                # set Rk to the distance of the second-to-last shell, for which we are sure to have full info
                Lk_temp = unique[-2]  # EX: 4
                # set k to cumulative up to last complete shell
                k_temp = index[-1]  # EX: 9

            # fix internal radius
            Ln_temp = np.rint(Lk_temp * ratio)
            # if the inner shell is accidently the same of the outer, go to the innerer one
            if Ln_temp == Lk_temp:
                Ln_temp -= 1

            n_temp = sum(dist_i <= Ln_temp)

            self.Lk[i] = Lk_temp
            self.k[i] = k_temp.astype(np.int64)
            self.n[i] = n_temp.astype(np.int64)
            self.Ln[i] = Ln_temp

        # checks out
        if any(~self.mask):
            print(
                "BE CAREFUL: "
                + str(sum(~self.mask))
                + " points would have Rk set to 0\
				   and thus will be kept out of the statistics. Consider increasing k."
            )

    # --------------------------------------------------------------------------------------

    def fix_k_shell(self, k_shell, ratio):

        """Computes the Lk, Ln, n given k_shell

        This routine computes the external radius Lk, the associated points k, internal radius Ln and associated points n.
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

        assert ratio > 0 and ratio < 1, "set a proper value for the ratio"

        self.Lk, self.k, self.Ln, self.n = (
            np.zeros(self.Nele),
            np.zeros(self.Nele, dtype=np.int64),
            np.zeros(self.Nele),
            np.zeros(self.Nele, dtype=np.int64),
        )
        self.mask = np.ones(self.Nele, dtype=bool)

        for i, dist_i in enumerate(self.distances):

            unique, index = np.unique(dist_i, return_index=True)

            # check whether the point has enough shells, at least one more than the one we want to consider
            if unique.shape[0] < k_shell + 1:
                self.mask[i] = False
                continue  # or Lk_temp = unique[-1] even if the shell is not the wanted one???

            # set Lk to the distance of the selected shell
            Lk_temp = unique[k_shell]
            # and Ln according to the ratio
            Ln_temp = np.rint(Lk_temp * ratio)
            # if the inner shell is accidently the same of the outer, go to the innerer one
            if Ln_temp == Lk_temp:
                Ln_temp -= 1

            # compute k and n
            if self.is_w:
                which_k = dist_i <= Lk_temp
                self.k[i] = sum(self.weights[self.dist_indices[i][which_k]]).astype(
                    np.int64
                )
                which_n = dist_i <= Ln_temp
                self.n[i] = sum(self.weights[self.dist_indices[i][which_n]]).astype(
                    np.int64
                )
            else:
                self.k[i] = index[k_shell + 1].astype(np.int64)
                self.n[i] = sum(dist_i <= Ln_temp).astype(np.int64)

            self.Lk[i] = Lk_temp
            self.Ln[i] = Ln_temp

        # checks out
        if any(~self.mask):
            print(
                "BE CAREFUL: "
                + str(sum(~self.mask))
                + " points would have Rk set to 0\
				   and thus will be kept out of the statistics. Consider increasing k."
            )

    # --------------------------------------------------------------------------------------

    def compute_id_binomial_k(self, k, shell=True, ratio=None):
        """Calculate Id using the binomial estimator by fixing the number of neighbours or shells

        As in the case in which one fix Lk, also in this version of the estimation
        one removes the central point from n and k. Two different ways of computing
        k,n,Lk,Ln are available, wheter one intends k as the k-th neighbour
        or the k-th shell. In both cases, one wants to be sure that the shell
        chosen is

        Args:
                k (int): order of neighbour that set the external shell
                ratio (float): ratio between internal and external shell

        """
        # checks-in and initialisations
        assert (
            self.distances is not None
        ), "first compute distances with the proper metric (manhattan of hamming presumably)"

        assert ratio > 0 and ratio < 1, "set a proper value for the ratio"

        if shell:
            assert k < self.maxk, "asking for a much too large number of shells"
            self.fix_k_shell(k, ratio)

        else:
            assert (
                k < self.maxk
            ), "You first need to recompute the distances with the proper number of NN"
            self.fix_k(k, ratio)

        n_eff = self.n[self.mask]
        k_eff = self.k[self.mask]

        E_n = n_eff.mean()
        if E_n == 1.0:
            print(
                "no points in the inner shell, returning 0. Consider increasing Lk and/or the ratio"
            )
            return 0

        if self.is_w:
            ww = self.weights[self.mask]
        else:
            ww = 1

        self.id_estimated_binom = SMin(    df.compute_binomial_logL, 
      args=(self.Lk[self.mask],k_eff-1,self.Ln[self.mask],n_eff-1,True,ww),
      bounds=(0.0001,100.),method='bounded'   ).x

    # ----------------------------------------------------------------------------------------------

    def set_id(self, d):
        assert d > 0, "cannot support negative dimensions (yet)"
        self.id_selected = d

    # ----------------------------------------------------------------------------------------------

    def set_Lk_Ln(self, lk, ln):
        assert (
            isinstance(
                ln,
                (
                    np.int64,
                    np.int32,
                    np.int16,
                    np.int8,
                    np.uint64,
                    np.uint32,
                    np.uint16,
                    np.uint8,
                    int,
                ),
            )
            and ln > 0
        ), "select a proper integer Ln>0"
        assert (
            isinstance(
                lk,
                (
                    np.int64,
                    np.int32,
                    np.int16,
                    np.int8,
                    np.uint64,
                    np.uint32,
                    np.uint16,
                    np.uint8,
                    int,
                ),
            )
            and lk > 0
        ), "select a proper integer Lk>0"
        assert lk > ln, "select Lk and Ln, s.t. Lk > Ln"
        self.Ln = ln
        self.Lk = lk

    # ----------------------------------------------------------------------------------------------

    def set_w(self, w):
        assert len(w) == self.Nele and all(
            [wi > 0 and isinstance(wi, (np.int, int))] for wi in w
        ), "load proper integer weights"
        self.weights = np.array(w, dtype=np.int)


# ----------------------------------------------------------------------------------------------


def _beta_prior_d(k, n, Lk, Ln, a0=1, b0=1, plot=True, verbose=True):
    """Compute the posterior distribution of d given the input aggregates
    Since the likelihood is given by a binomial distribution, its conjugate prior is a beta distribution.
    However, the binomial is defined on the ratio of volumes and so do the beta distribution. As a
    consequence one has to change variable to have the distribution over d

    Args:
            k (nd.array(int)): number of points within the external shells
            n (nd.array(int)): number of points within the internal shells
            Lk (int): outer shell radius
            Lk (int): inner shell radius
            a0 (float): beta distribution parameter, default =1 for flat prior
            b0 (float): prior initialiser, default =1 for flat prior
            plot (bool, default=False): plot the posterior
    Returns:
            E_d_emp (float): mean value of the posterior
            S_d_emp (float): std of the posterior
            d_range (ndarray(float)): domain of the posterior
            P (ndarray(float)): probability of the posterior
    """
    from scipy.special import beta as beta_f
    from scipy.stats import beta as beta_d

    a = a0 + n.sum()
    b = b0 + k.sum() - n.sum()
    posterior = beta_d(a, b)

    def p_d(d):
        Vk = df.compute_discrete_volume(Lk, d)
        Vn = df.compute_discrete_volume(Ln, d)
        dVk_dd = df.compute_derivative_discrete_vol(Lk, d)
        dVn_dd = df.compute_derivative_discrete_vol(Ln, d)
        x = Vn / Vk
        dx_dd = dVn_dd / Vk - dVk_dd * Vn / Vk / Vk
        return abs(posterior.pdf(x) * dx_dd)

    dx = 0.1
    d_left = 0.000001
    d_right = 20 + dx + d_left
    d_range = np.arange(d_left, d_right, dx)
    P = np.array([p_d(di) for di in d_range]) * dx
    counter = 0
    elements = sum(P != 0)
    while elements < 1000:
        if elements > 10:
            dx /= 10
            ind = np.where(P != 0)[0]
            d_left = d_range[ind[0]]
            d_right = d_range[ind[-1]]
        else:
            dx /= 10

        d_range = np.arange(d_left, d_right, dx)
        P = np.array([p_d(di) for di in d_range]) * dx
        elements = sum(P != 0)
        counter += 1
        if verbose:
            print("iter no\t", counter, d_left, d_right, elements)

    P = P.reshape(P.shape[0])

    if plot:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(d_range, P)
        plt.xlabel("d")
        plt.ylabel("P(d)")
        plt.title("posterior of d")
        plt.show()

    E_d_emp = (d_range * P).sum()
    S_d_emp = np.sqrt((d_range * d_range * P).sum() - E_d_emp * E_d_emp)
    print("empirical average:\t", E_d_emp, "\nempirical std:\t\t", S_d_emp)
    # 	theoretical results, valid only in the continuum case
    # 	E_d = ( sp.digamma(a) - sp.digamma(a+b) )/np.log(r)
    # 	S_d = np.sqrt( ( sp.polygamma(1,a) - sp.polygamma(1,a+b) )/np.log(r)**2 )

    return E_d_emp, S_d_emp, d_range, P


# --------------------------------------------------------------------------------------

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
