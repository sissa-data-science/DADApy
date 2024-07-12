# Copyright 2021 The DADApy Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
The *id_discrete* module contains the *IdDiscrete* class.

The different algorithms of intrinsic dimension estimation for discrete spaces
  are implemented as methods of this class.
"""

import multiprocessing

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp as KS

import dadapy._utils.discrete_functions as df
from dadapy import plot as ddp
from dadapy.base import Base

cores = multiprocessing.cpu_count()
rng = np.random.default_rng()


class IdDiscrete(Base):
    """Estimates the intrinsic dimension of a dataset with discrete features using a binomial likelihood.

    Inherits from class Base.

    Attributes:
        lk (int, np.ndarray(int)): radius of the external shells
        ln (int, np.ndarray(int)): radius of the internal shells
        k (np.ndarray(int)): total number of points within the external shell
        n (np.ndarray(int)): total number of points within the internal shell
        ratio (float): ratio between internal and external radii
        intrinsic_dim (float): intrinsic dimension obtained with the binomial estimator
        intrinsic_dim_err (float): error associated with the id estimation.
            Computed through Cramer-Rao or Bayesian inference
        intrinsic_dim_scale (float): scale at which the id has been computed
        posterior_domain (np.ndarray(float)): eventual support of the posterior distribution of the id
        posterior (np.ndarray(float)): posterior distribution when evaluated with Bayesian inference
    """

    def __init__(
        self,
        coordinates=None,
        distances=None,
        is_network=False,
        maxk=None,
        condensed=None,
        weights=None,
        verbose=False,
        n_jobs=cores,
    ):
        """Instantiate the IdDiscrete object.

        Args:
            coordinates (np.ndarray(float)): the data points loaded, of shape (N , dimension of embedding space)
            distances (np.ndarray(float)): A matrix of dimension N x mask containing distances between points
            is_network (bool, default=False): if True a network is assumed to be loaded, meaning that in each
                computation the central points must not be removed from the enumeration
            maxk (int): maximum number of neighbours to be considered for the calculation of distances
            condensed (bool, default=False): if True, distances are saved in a cumulative fashion: the position
                in the self.distances array is the distance and the number represents the amount of points up to that
                distance (self is included)
            weights (np.ndarray(int), default=None): keeps into account explicitly possible repetitions of datapoints
            verbose (bool): whether you want the code to speak or shut up
            n_jobs (int): number of cores to be used

        """
        super().__init__(
            coordinates=coordinates,
            distances=distances,
            maxk=maxk,
            verbose=verbose,
            n_jobs=n_jobs,
        )

        self.central_point = 0 if is_network else 1
        if weights is None:
            self._is_w = False
            self._weights = None
        else:
            self.set_w(weights)
            self._is_w = True

        self.lk = None
        self.ln = None
        self.k = None
        self.n = None
        self.ratio = None

        self._k = None
        self._mask = None
        if distances is not None:
            self._condensed = False
        else:
            self._condensed = condensed

        self.intrinsic_dim = None
        self.intrinsic_dim_err = None
        self.intrinsic_dim_scale = None
        self.posterior_domain = None
        self.posterior = None

    # ----------------------------------------------------------------------------------------------

    def compute_distances(
        self, maxk=None, metric="manhattan", period=None, condensed=None, d_max=100
    ):
        """Compute distances between datapoints. Distances can be saved in an extended or compacted shape.

        If condensed is True, self.distances will store how many points are present up to that distance.
        Conversely, the usual scheme is adopted (see same method in Base)

        Args:
            maxk: maximum number of neighbours for which distance is computed and stored
            metric: type of metric
            period (float or np.ndarray): periodicity (only used for periodic distance computation). Default is None.
            condensed (bool): save how many points one finds at each distance instead of all distances
            d_max (int, default=100): decide the farthest distance to be saved (only if condensed is True)

        """
        if condensed is not None:
            self._condensed = condensed
        else:
            assert (
                self._condensed is not None
            ), 'initialize "condensed" parameter in init or compute distances'

        if self._condensed:
            self.metric = metric
            self.period = period
            if period is not None:
                if isinstance(period, np.ndarray) and period.shape == (self.dims,):
                    self.period = np.array(period, dtype=float)
                elif isinstance(period, int):
                    self.period = np.full(self.dims, fill_value=period, dtype=float)
                else:
                    raise ValueError(
                        f"'period' must be either a float scalar or a numpy array of floats of shape ({self.dims},)"
                    )

            if ~isinstance(self.X[0, 0], int):
                print(
                    "N.B. the data will be passed to the routine to compute distances as integers"
                )

            self.distances, self.dist_indices = df.return_condensed_distances(
                np.array(self.X, dtype=int),
                self.metric,
                d_max,
                self.period,
                self.n_jobs,
            )

        else:
            super().compute_distances(maxk, metric, period)

    # ----------------------------------------------------------------------------------------------

    def fix_lk(self, lk=None, ln=None):
        """Compute the k points within the given Rk and n points within given Rn.

        For each point, computes the number self.k of points within a sphere of radius Rk
        and the number self.n within an inner sphere of radius Rk. It also provides
        a mask to take into account those points for which the statistics might be wrong, i.e.
        k == self.maxk, meaning that all available points were selected or when we are not
        sure to have all the point of the selected shell. If self.maxk is equal
        to the number of points of the dataset no mask will be applied, as all the point
        were surely taken into account.
        The mask is not necessary (i.e. is set to True) if condensed is True, as all points will
        be considered.
        Eventually add the weights of the points if they have an explicit multiplicity

        Args:
            lk (int): external shell radius
            ln (int): internal shell radius

        """
        # checks-in and initializations
        assert (
            self.distances is not None
        ), "first compute distances with the proper metric (manhattan of hamming presumably)"

        if lk is not None and ln is not None:
            self.set_lk_ln(lk, ln)
        else:
            assert (
                self.lk is not None and self.ln is not None
            ), "set lk and ln or insert proper values for the lk and ln parameters"
        self.set_ratio(float(self.ln) / float(self.lk))

        self._k = (
            0  # just a flag to remember that the id was computed at constant radius
        )

        if self._condensed:
            self.n = np.copy(self.distances[:, self.ln])
            self.k = np.copy(self.distances[:, self.lk])
            self._mask = np.ones(self.N, dtype=bool)

        else:
            if self._is_w is False:
                self.k = (self.distances <= self.lk).sum(axis=1)
                self.n = (self.distances <= self.ln).sum(axis=1)
            else:
                assert (
                    self._weights is not None
                ), "first insert the weights if you want to use them!"
                self.k = np.array(
                    [
                        sum(self._weights[self.dist_indices[i][el]])
                        for i, el in enumerate(self.distances <= self.lk)
                    ],
                    dtype=np.int,
                )
                self.n = np.array(
                    [
                        sum(self._weights[self.dist_indices[i][el]])
                        for i, el in enumerate(self.distances <= self.ln)
                    ],
                    dtype=np.int,
                )

            # checks-out
            # compute mask
            if self.maxk == self.N - 1:
                self._mask = np.ones(self.N, dtype=bool)
            else:
                # if not all possible NN were taken into account (i.e. maxk < N) and k is equal to self.maxk
                # or distances[:,-1]<lk, it is likely that there are other points within lk that are not being
                # considered and thus potentially altering the statistics -> neglect them through self._mask
                # in the calculation of likelihood
                self._mask = self.distances[:, -1] > self.lk  # or self.k == self.maxk

                if np.any(~self._mask):
                    print(
                        "NB: for "
                        + str(sum(~(self._mask)))
                        + " points, the counting of k could be wrong, "
                        + "as more points might be present within the selected Rk. In order not to affect "
                        + "the statistics a mask is provided to remove them from the calculation of the "
                        + "likelihood or posterior.\nConsiself.kder recomputing NN with higher maxk or lowering Rk."
                    )
        if self.verb:
            print("n and k computed")

    # ----------------------------------------------------------------------------------------------

    def compute_id_binomial_lk(  # noqa: C901
        self,
        lk=None,
        ln=None,
        method="bayes",
        subset=None,
        plot=True,
        set_attr=True,
    ):
        """Calculate Id using the binomial estimator by fixing the eternal radius for all the points.

        In the estimation of d one has to remove the central point from the counting of n and k
        as it generates the process, but it is not effectively part of it

        Args:
            lk (int): radius of the external shell
            ln (int): radius of the internal shell
            subset (int): choose a random subset of the dataset to make the Id estimate
            method (str, default 'bayes'): choose method between 'bayes' and 'mle'. The bayesian estimate
                    gives the distribution of the id, while mle only the max of the likelihood
            plot (bool): if bayes method is used, one can decide whether to plot the posterior
            set_attr (bool): changes class attributes after computation

        Returns:
            intrinsic_dim (float): the id esteem
            intrinsic_dim_err (float): the error estimate on the id
            intrinsic_dim_scale (float): the scale at which the id was computed (lk)

        """
        # checks-in and initialisations
        assert (
            self.distances is not None
        ), "first compute distances with the proper metric (manhattan of hamming presumably)"

        if lk is not None and ln is not None:
            if isinstance(self.lk, np.ndarray):  # id previously computed by fixing k
                self.set_lk_ln(lk, ln)
                self.fix_lk()
            elif (
                lk != self.lk or ln != self.ln
            ):  # ln or lk changing from a previous estimation
                self.set_lk_ln(lk, ln)
                self.fix_lk()
        else:
            assert (
                self.lk is not None and self.ln is not None
            ), "set lk and ln through set_lk_ln or insert proper values for the lk and ln parameters"

        mask = self._my_mask(subset)
        n_eff = self.n[mask] - self.central_point
        k_eff = self.k[mask] - self.central_point

        if self._is_w:
            w_eff = self._weights[mask]

        # check statistics before performing id estimation
        if ~self._is_w:
            e_n = n_eff.mean()
            if e_n == 0.0:
                print(
                    "no points in the inner shell, returning 0. Consider increasing ln and possibly lk"
                )
                self.intrinsic_dim = 0
                self.intrinsic_dim_err = 0
                return 0, 0, self.lk

        # choice of the method
        if method == "mle":
            if self._is_w:  # necessary only if using the root finding method
                N = w_eff.sum()
                k_eff = sum(k_eff * w_eff) / float(N)
                n_eff = sum(n_eff * w_eff) / float(N)
            else:
                w_eff = 1
                N = k_eff.shape[0]
                k_eff = k_eff.mean()
                n_eff = n_eff.mean()

            # explicit computation of likelihood, not necessary when ln and lk are fixed, but apparently\
            # more stable than root searching
            intrinsic_dim = df.find_d_likelihood(self.ln, self.lk, n_eff, k_eff, w_eff)
            """
            intrinsic_dim = df.find_d_root(self.ln, self.lk, n_eff, k_eff)
            """
            intrinsic_dim_err = (
                df.binomial_cramer_rao(
                    d=intrinsic_dim, ln=self.ln, lk=self.lk, N=N, k=k_eff
                )
                ** 0.5
            )
        elif method == "bayes":
            if self._is_w:
                k_tot = w_eff * k_eff
                n_tot = w_eff * n_eff
            else:
                k_tot = k_eff
                n_tot = n_eff

            if self.verb:
                print("starting bayesian estimation")

            (
                intrinsic_dim,
                intrinsic_dim_err,
                self.posterior_domain,
                self.posterior,
            ) = df.beta_prior_d(k_tot, n_tot, self.lk, self.ln, plot=plot)
        else:
            print("select a proper method for id computation")
            return 0

        if set_attr:
            self.intrinsic_dim = intrinsic_dim
            self.intrinsic_dim_err = intrinsic_dim_err
            self.intrinsic_dim_scale = self.lk

        return intrinsic_dim, intrinsic_dim_err, self.lk

    # ----------------------------------------------------------------------------------------------

    def return_id_scaling(self, Lks, r=0.5, method="mle", subset=None, plot=True):
        """Compute the ID by varying the radii.

        Args:
            Lks (list or np.ndarray(int)): external radii lk
            r (float, default = 0.5): ratio among ln and lk
            method (string, default='mle'): method to compute the id
            subset (np.ndarray(int) or int, default=None): indices of points to be used for the estimate
            plot (bool, default=True): whether to plot the id scaling

        Returns:
            ids (np.ndarray): intrinsic dimension at different scales
            ids_err (np.ndarray): error estimate on intrinsic dimension at different scales

        Quick Start:
        ===========

        .. code-block:: python

                import numpy as np
                import matplotlib.pyplot as plt
                rng = np.random.default_rng(12345)
                from dadapy.id_discrete import IdDiscrete

                #uniformly sampled points on 5d lattice

                N = 2500
                box = 50
                d = 5
                data = rng.integers(0,box,size=(N, d))

                I3D = IdDiscrete(data, condensed=True, maxk=data.shape[0])
                I3D.compute_distances(metric='manhattan',d_max=box*d,period=box)

                scales = np.arange(15,30)

                ids_scaling, ids_scaling_err = I3D.return_id_scaling(scales,r=0.75)

                ids_scaling:
                array([4.91, 4.86, 4.86, 4.94, 4.98, 5.  , 5.  , 4.99, 5.01, 5.  , 5.01, 5.01, 5.02, 5.01, 5.01])

                ids_scaling_err:
                array([0.09, 0.08, 0.07, 0.06, 0.05, 0.05, 0.04, 0.04, 0.03, 0.03, 0.03, 0.02, 0.02, 0.02, 0.02])

        References:
            Iuri Macocco, Aldo Glielmo, Jacopo Grilli, and Alessandro Laio, Intrinsic Dimension Estimation for Discrete
            Metrics, Phys. Rev. Lett. 130, 067401 – Published 8 February 2023

        """
        ids = np.zeros_like(Lks, dtype=float)
        ids_e = np.zeros_like(Lks, dtype=float)

        for i, lk in enumerate(Lks):
            ln = np.ceil(lk * r).astype(int)
            if ln == lk:
                ln -= 1

            id_i, id_er, scale = self.compute_id_binomial_lk(
                lk, ln, method=method, subset=subset, set_attr=False
            )
            ids[i] = id_i
            ids_e[i] = id_er

        if plot:
            plt.figure()
            plt.plot(Lks, ids)
            plt.errorbar(Lks, ids, ids_e, fmt="None")
            plt.xlabel("scale")
            plt.ylabel("ID estimate")

        return ids, ids_e

    # ----------------------------------------------------------------------------------------------

    def fix_k(self, k_eff=None, ratio=0.5):  # noqa: C901
        """Compute Rk, Rn, n for each point given a selected value of k.

        This routine computes external radii lk, internal radii ln and internal points n.
        It also ensures that lk is scaled onto the most external shell
        for which we are sure to have all points.
        A mask will exclude "badly behaved" points if condensed is False.
        NB As a consequence the k will be point dependent

        Args:
            k_eff (int, default=self.k_max): selected (max) number of NN
            ratio (float, default = 0.5): approximate ratio among ln and lk

        """
        # TODO: what if we have multiplicity???? use it straightly on the plain points
        #       or take it into account when counting the number of NN?

        # checks-in and initialisations
        assert (
            self.distances is not None
        ), "first compute distances with the proper metric (manhattan of hamming presumably)"

        self.set_ratio(ratio)
        self._k = k_eff  # mark that the id is computed at fixed k

        if self._condensed:
            self.lk = np.ones(self.N, dtype=int)
            for i, ddi in enumerate(self.distances):
                lk = 0
                appo = 0
                while (
                    k_eff > ddi[lk + 1]
                ):  # increase lk until you reach the proper amount of neighbours
                    if ddi[lk + 1] != ddi[lk]:
                        # save at which range you last found a neighbour at a different distance
                        appo = np.copy(lk) + 1
                    lk += 1

                # go to next radius if it has exactly the number of k_eff
                if ddi[lk + 1] == k_eff:
                    lk += 1
                    self.lk[i] = np.copy(lk)
                    continue
                # go back to lower radius with same amount of neighbours if the next one is too big
                if ddi[lk] == ddi[appo]:
                    self.lk[i] = np.copy(appo)
                else:
                    self.lk[i] = np.copy(lk)

            self.ln = np.rint(self.lk * self.ratio).astype(int)
            self.k = np.take_along_axis(self.distances, self.lk[:, None], 1)[:, 0]
            self.n = np.take_along_axis(self.distances, self.ln[:, None], 1)[:, 0]
            self._mask = self.lk > 0

        else:
            if k_eff is None:
                k_eff = self.maxk - 1
            else:
                assert k_eff < self.maxk, (
                    "A k_eff > maxk was selected, recompute the distances with the proper amount on NN to see points "
                    "up to that k_eff "
                )

            # routine
            self.lk, self.k, self.ln, self.n = (
                np.zeros(self.N),
                np.zeros(self.N, dtype=np.int64),
                np.zeros(self.N),
                np.zeros(self.N, dtype=np.int64),
            )
            self._mask = np.ones(self.N, dtype=bool)

            # cut distances at the k-th NN and cycle over each point
            for i, dist_i in enumerate(self.distances[:, : k_eff + 1]):
                # case 1: all possible neighbours have been considered, no extra possible unseen points
                if k_eff == self.N:
                    lk_temp = dist_i[-1]
                    k_temp = k_eff

                # case 2: only some NN are considered -> work on surely "complete" shells
                else:
                    # Group distances according to shell. Returns radii of shell and cumulative number
                    # of points at that radius.
                    # The index at which a distance is first met == number of points at smaller distance
                    # EXAMPLE:
                    """
                    a = np.array([0,1,1,2,3,3,4,4,4,6,6])     and suppose it would go on as 6,6,6,7,8...
                    un, ind = np.unique(a, return_index=True)
                    un: array([0, 1, 2, 3, 4, 6])
                    ind: array([0, 1, 3, 4, 6, 9])
                    """
                    unique, index = np.unique(dist_i, return_index=True)

                    if unique.shape[0] < 3:
                        # lk=0 may happen when we have smtg like dist_i = [0,0,0,0,0] or dist_i = [0,3,3,3,3,3].
                        # As we don't have the full information for at least two shells, we skip the point
                        self._mask[i] = False
                        continue

                    # set lk to the distance of the second-to-last shell, for which we are sure to have full info
                    lk_temp = unique[-2]  # EX: 4
                    # set k to cumulative up to last complete shell
                    k_temp = index[-1]  # EX: 9

                # fix internal radius
                ln_temp = np.rint(lk_temp * self.ratio)
                # if the inner shell is accidentally the same of the outer, go to the innerer one
                if ln_temp == lk_temp:
                    ln_temp -= 1

                n_temp = sum(dist_i <= ln_temp)

                self.k[i] = k_temp.astype(np.int64)
                self.n[i] = n_temp.astype(np.int64)
                self.lk[i] = lk_temp
                self.ln[i] = ln_temp

            # checks out
            if any(~self._mask):
                print(
                    "BE CAREFUL: "
                    + str(sum(~self._mask))
                    + " points would have lk set to 0 "
                    + "and thus will not be considered when computing the id. Consider increasing k."
                )

        if self.verb:
            print("n and k computed")

    # --------------------------------------------------------------------------------------

    def fix_k_shell(self, k_shell, ratio=0.5):  # noqa: C901
        """Compute the lk, ln, n given k_shell.

        This routine computes the external radius lk, the associated points k,
        internal radius ln and associated points n. The computation is performed
        starting from a given number of shells. It ensures that Rk is scaled onto
        the most external shell for which we are sure to have all points.
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

        self.set_ratio(ratio)

        if self._condensed:
            assert k_shell < self.distances.shape[1]
            self.lk = np.zeros(self.N, dtype=int)
            for i, ddi in enumerate(self.distances):
                counter = 0  # number of non-empty shell visited
                lk = 1  # radius iterator
                while counter < k_shell:
                    if ddi[lk - 1] != ddi[lk]:
                        counter += 1
                    lk += 1

                self.lk[i] = np.copy(lk) - 1

            self.ln = np.rint(self.lk * self.ratio).astype(int)
            # check whether ln==lk, in that case reduce ln by 1
            mask_ll = np.where(self.lk == self.ln)[0]
            self.ln[mask_ll] -= 1

            self.k = np.take_along_axis(self.distances, self.lk[:, None], 1)[:, 0]
            self.n = np.take_along_axis(self.distances, self.ln[:, None], 1)[:, 0]
            self._mask = np.ones(self.N, dtype=bool)

        else:
            assert k_shell < self.maxk, "asking for a much too large number of shells"

            self.lk, self.k, self.ln, self.n = (
                np.zeros(self.N),
                np.zeros(self.N, dtype=np.int64),
                np.zeros(self.N),
                np.zeros(self.N, dtype=np.int64),
            )
            self._mask = np.ones(self.N, dtype=bool)

            for i, dist_i in enumerate(self.distances):
                counter = 0
                cycle = 0
                while counter < k_shell + 1:
                    if dist_i[cycle] != dist_i[cycle + 1]:
                        counter += 1  # update shell counter when neighbours at different distance are found
                    cycle += 1
                    if cycle == self.maxk:  # if you reach maxk do not use the point
                        self._mask[i] = False
                        continue

                self.k[i] = np.copy(cycle)
                lk_temp = np.copy(
                    dist_i[cycle - 1]
                )  # set lk to the distance of the selected shell
                # and ln according to the ratio
                ln_temp = np.rint(lk_temp * self.ratio)
                # if the inner shell is accidentally the same of the outer, go to the innerer one
                if ln_temp == lk_temp:
                    ln_temp -= 1

                # compute k and n
                if self._is_w:
                    which_k = dist_i <= lk_temp
                    self.k[i] = sum(
                        self._weights[self.dist_indices[i][which_k]]
                    ).astype(np.int64)
                    which_n = dist_i <= ln_temp
                    self.n[i] = sum(
                        self._weights[self.dist_indices[i][which_n]]
                    ).astype(np.int64)
                else:
                    self.n[i] = sum(dist_i <= ln_temp).astype(np.int64)

                self.lk[i] = lk_temp
                self.ln[i] = ln_temp

            # checks out
            if any(~self._mask):
                print(
                    "BE CAREFUL: "
                    + str(sum(~self._mask))
                    + " points would have Rk set to 0 "
                    + "and thus will be kept out of the statistics. Consider increasing k."
                )
        self._k = k_shell
        if self.verb:
            print("n and k computed")

    # --------------------------------------------------------------------------------------

    def compute_id_binomial_k_discrete(
        self, k, ratio=0.5, shell=False, subset=None, approx_err=True, set_attr=True
    ):
        """Calculate Id using the binomial estimator by fixing the number of neighbours or shells.

        As in the case in which one fix lk, also in this version of the estimation
        one removes the central point from n and k. Two different ways of computing
        k,n,lk,ln are available, whether one intends k as the k-th neighbour
        or the k-th shell. In both cases, one wants to be sure that the shell
        chosen is

        Args:
            k (int): order of neighbour that set the external shell
            ratio (float): ratio between internal and external shell
            shell (bool): k stands for number of neighbours or number of occupied shells
            subset (int): choose a random subset of the dataset to make the Id estimate
            approx_err (bool, default=True): if True, computes the error on the estimate using the CR.\
                Otherwise it profiles the likelihood and compute its std (takes longer)
            set_attr (bool, default=True): assign id, id error and scale to the class

        Returns:
            intrinsic_dim (float): the id estimation
            intrinsic_dim_err (float): the error estimate on the id
            intrinsic_dim_scale (float): the scale at which the id was computed: <lk>

        """
        # checks-in and initialisations
        assert (
            self.distances is not None
        ), "first compute distances with the proper metric (manhattan of hamming presumably)"

        if k != self._k or ratio != self.ratio:
            if shell:
                self.fix_k_shell(k, ratio)
            else:
                self.fix_k(k, ratio)

        mask = self._my_mask(subset)

        n_eff = self.n[mask] - self.central_point
        k_eff = self.k[mask] - self.central_point
        ln_eff = self.ln[mask]
        lk_eff = self.lk[mask]
        if self._is_w:
            ww = self._weights[mask]
        else:
            ww = 1

        e_n = n_eff.mean()
        if e_n == 0.0:
            print(
                "no points in the inner shell, returning 0. Consider increasing k and/or the ratio,"
                "especially if your embedding space is very high dimensional."
            )
            self.intrinsic_dim = 0
            self.intrinsic_dim_err = 0
            return 0, 0, lk_eff.mean()

        intrinsic_dim = df.find_d_likelihood(ln_eff, lk_eff, n_eff, k_eff, ww)
        if approx_err:
            intrinsic_dim_err = (
                df.binomial_cramer_rao(
                    d=intrinsic_dim,
                    ln=int(ln_eff.mean()),
                    lk=int(lk_eff.mean()),
                    N=mask.sum(),
                    k=k_eff.mean(),
                )
                ** 0.5
            )
        else:
            _, b, _, _ = df.profile_likelihood(ln_eff, lk_eff, n_eff, k_eff, ww)
            intrinsic_dim_err = b

        if set_attr:
            self.intrinsic_dim = intrinsic_dim
            self.intrinsic_dim_err = intrinsic_dim_err
            self.intrinsic_dim_scale = lk_eff.mean()

        return intrinsic_dim, intrinsic_dim_err, lk_eff.mean()

    # ----------------------------------------------------------------------------------------------

    def return_id_scaling_k(self, Ks, r=0.5, subset=None, plot=True):
        """Compute the ID by varying the number of neighbours considered.

        Args:
            Ks (list or np.ndarray(int)): number of neighbours
            r (float, default = 0.5): ratio between ln and lk
            subset (int): choose a random subset of the dataset to make the Id estimate
            plot (bool, default=True): whether to plot the id as a function of Ks

        Returns:
            ids (np.ndarray(float)): intrinsic dimension at different scales
            ids_err (np.ndarray(float)): error estimate on intrinsic dimension at different scales
            scale (np.ndarray(float)): scale at which the id was computed (mean values of lk)

        """
        ids = np.zeros_like(Ks, dtype=float)
        ids_e = np.zeros_like(Ks, dtype=float)
        scale = np.zeros_like(Ks, dtype=float)

        for i, k in enumerate(Ks):
            ids[i], ids_e[i], scale[i] = self.compute_id_binomial_k_discrete(
                k, r, False, subset=subset, set_attr=False
            )

        if plot:
            plt.plot(Ks, ids)
            plt.errorbar(Ks, ids, ids_e, fmt="None")
            plt.xlabel("scale")
            plt.ylabel("ID estimate")

        return ids, ids_e, scale

    # ----------------------------------------------------------------------------------------------

    def return_id_fit(self, scales, subset=None, plot=True):
        r"""Find the ID as least square fit between the actual and the theoretical number of points.

        Compute the id as least square fit between the average number of neighbours N(r) and the theoretical number
        of points within a given shell \rho*V(r). Given a radius r, the fit takes into account all points with r_i <= r.
        The fit is performed both between N and \rho*V and their logarithms.

        Args:
            scales (list(int) or np.ndarray(int)): radii at which the number of neighbours is computed
            subset (np.ndarray(int)): indices of points to be used for the estimate
            plot (bool): plot the fit taking into account all given radii in R and ID(r)

        Returns:
            ids (np.ndarray(float)): the intrinsic dimension estimates as a function of R with "normal" fit
            ids (np.ndarray(float)): the intrinsic dimension estimates as a function of R with "log" fit
            scales (np.ndarray(int)): the scales at which the ID is computed (the first point is excluded)

        """
        from scipy.optimize import curve_fit

        def fit_func(r, d, rho):
            return df.compute_discrete_volume(r, d) * rho

        def fit_func_log(r, d, rho):
            return np.log(df.compute_discrete_volume(r, d)) + np.log(rho)

        ids, idsl = [], []
        N = np.zeros_like(scales)
        mask = self._my_mask(subset)

        for i, ri in enumerate(scales):
            if self._condensed:
                N[i] = np.mean(self.distances[mask, ri])
            else:
                N[i] = np.mean(self.distances[mask] <= ri)

            if i == 0:
                continue

            popt, pcov = curve_fit(fit_func, scales[: i + 1], N[: i + 1])
            ids.append(popt)
            poptl, pcovl = curve_fit(fit_func_log, scales[: i + 1], np.log(N[: i + 1]))
            idsl.append(poptl)

        ids = np.array(ids)
        idsl = np.array(idsl)

        if plot:
            plt.figure()
            plt.plot(N, fit_func(scales, *popt), label="fit")
            plt.plot(N, np.exp(fit_func_log(scales, *poptl)), label="log fit")
            plt.scatter(N, N, label="x=y")
            plt.legend()

            plt.figure()
            plt.plot(scales[1:], ids[:, 0], label="fit")
            plt.plot(scales[1:], idsl[:, 0], label="log fit")
            plt.legend()

        return ids[:, 0], idsl[:, 0], scales[1:]

    # ----------------------------------------------------------------------------------------------

    def return_id_fit_continuum(self, R, fit_intercept=True, subset=None, plot=True):
        r"""Find the ID as linear fit between Log(n) vs Log(r), assuming N~r**d.

        Compute the id fitting a line between Log(n) and Log(r). Given a radius r_i, the fit takes into account
        all points with r <= r_i.
        Since no prescription has been given for this method on how to deal with points lying at discrete distances,
        we suppose to apply a small smearing, so that half of the points will end up a bit further than their initial
        distance and half will be slightly closer. As a consequence, when looking at distance r, we will be considering
        all points up to r-1 and half of those at distance r

        Args:
            R (list(int) or np.ndarray(int)): radii at which the number of neighbours is computed
            fit_intercept (bool, default=True): decide whether to fit the intercept or fix it to N(r=1)
            subset (np.ndarray(int)): indices of points to be used for the estimate
            plot (bool): plot the fit taking into account all given radii in R and ID(r)

        Returns:
            ids (np.ndarray(float)): the intrinsic dimension estimates as a function of R
            R (np.ndarray(int)): the scales at which the ID is computed (the first point is excluded)

        """
        self._mask = np.ones(self.N, dtype=bool)
        mask = self._my_mask(subset)
        el = mask.sum()

        from scipy.optimize import curve_fit

        if fit_intercept:

            def fit_func(x, a, b):
                return a * x + b

        else:
            if self._condensed:
                b = np.log(np.mean(self.distances[mask, 1])) - el
            else:
                b = np.log(np.mean(self.distances[mask] <= 1)) - el

            def fit_func(x, a):
                return a * x + b

        N = []
        ids_i = []

        if np.any(R < 1e-10):
            print("we cannot use the R=0 as we have to take the log, start with R=1")

        for i, r in enumerate(R):
            if self._condensed:
                shell = (
                    np.sum(self.distances[mask, r] - self.distances[mask, r - 1]) * 0.5
                )
                n = np.sum(self.distances[mask, r - 1]) + shell - el
            else:
                shell = (
                    np.sum((r - 1 < self.distances[mask]) * (self.distances[mask] <= r))
                    * 0.5
                )
                n = np.sum(self.distances[mask] < r) + shell - el

            N.append(n)

            if i == 0:
                continue
            popt, pcov = curve_fit(fit_func, np.log(R[: i + 1]), np.log(N[: i + 1]))
            ids_i.append(popt)

        ids = np.array(ids_i)

        if plot:
            plt.figure(figsize=(5, 3))
            plt.scatter(np.log(R), np.log(N), label="points")
            plt.plot(np.log(R), fit_func(np.log(R), *ids[-1]), label="linear fit")
            plt.legend()

            plt.figure(figsize=(5, 3))
            plt.plot(R[1:], ids[:, 0], label="curve_fit")

        return ids[:, 0], R[1:]

    # -------------------------------MODEL VALIDATION-----------------------------------------------
    # ----------------------------------------------------------------------------------------------

    def compute_local_density(self, plot=True):
        """Compute and compare the rescaled local density of points.

        Compute and compare the local density of points inside inner and outer shells
        after fixing lk and ln as n/<n> and (k-n)/<k-n>.
        The closest the points to the y=x line, the more the local uniform density
        hypothesis is respected, the better will be the id estimate

        Args:
            plot (bool, default=True): decide whether to plot the local density
        Returns:
            n (np.ndarray(float)): n/<n>
            m (np.ndarray(float)): (k-n)/<k-n>

        """
        assert self.n is not None and self.k is not None, "find k and n first!"
        assert self.intrinsic_dim is not None, "compute the ID first!"

        n = np.copy(self.n).astype(float)
        m = np.copy(self.k - self.n).astype(float)

        # id computed at fixed lk
        if self._k == 0:
            assert isinstance(self.lk, int)
            n /= n.mean()
            m /= m.mean()

        # id computed at fixed k
        else:
            Vk = df.compute_discrete_volume(self.lk, self.intrinsic_dim)
            Vn = df.compute_discrete_volume(self.ln, self.intrinsic_dim)
            n /= Vn
            m /= Vk - Vn

        dist = abs(n - m) / np.sqrt(n**2 + m**2)
        print("average distance from line: ", dist.mean())

        if plot:
            plt.scatter(n, m, s=1.0)
            plt.plot(
                np.linspace(min(min(n), min(m)), max(max(n), max(m)), 100),
                np.linspace(min(min(n), min(m)), max(max(n), max(m)), 100),
            )
            plt.xlabel(r"$n/\langle n \rangle$")
            plt.ylabel(r"$(k-n)/\langle k-n \rangle$")

        return n, m

    # ----------------------------------------------------------------------------------------------

    def model_validation_full(  # noqa: C901
        self,
        alpha=0.05,
        subset=None,
        artificial_samples=100000,
        pdf=False,
        cdf=True,
        filename=None,
    ):
        """Use Kolmogorov-Smirnoff test to assess the goodness of the id estimate.

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
            alpha (float, default=0.05): tolerance for the KS test
            subset (int or np.ndarray(int)): subset of points used to perform the model validation
            artificial_samples (int, default=100000): number of points sampled from the theoretical distribution
            pdf (bool, default=False): plot histogram of n_emp and n_i
            cdf (bool, default=False): plot cdf of n_emp and n_i
            filename (str, default=None): directory where to save plots
        Returns:
            s (float): KS statistics, max distance between empirical and theoretical cdfs
            pv (float): p-value associated to the KS statistics

        """
        if self.intrinsic_dim is None:
            print("compute the id before validating the model!")
            return 0

        mask = self._my_mask(subset)
        n_eff = self.n[mask] - self.central_point
        k_eff = self.k[mask] - self.central_point

        if isinstance(self.ln, np.ndarray):  # id estimated at fixed K
            ln_eff = self.ln[mask]
            lk_eff = self.lk[mask]

            title = "K=" + str(self._k)

            p = df.compute_discrete_volume(
                ln_eff, self.intrinsic_dim
            ) / df.compute_discrete_volume(lk_eff, self.intrinsic_dim)

        else:  # id estimated at fixed lk
            title = "R=" + str(self.lk)

            p = df.compute_discrete_volume(
                self.ln, self.intrinsic_dim
            ) / df.compute_discrete_volume(self.lk, self.intrinsic_dim)

        if self.n.shape[0] < artificial_samples:
            replicas = int(artificial_samples / self.n.shape[0])
            if isinstance(self.ln, np.ndarray):
                n_model = np.array(
                    [rng.binomial(ki, pi, size=replicas) for ki, pi in zip(k_eff, p)]
                ).reshape(-1)
            else:
                n_model = np.array(
                    [rng.binomial(ki, p, size=replicas) for ki in k_eff]
                ).reshape(-1)

        else:
            n_model = rng.binomial(k_eff, p)

        if self.central_point == 0:
            n_model = n_model[n_model > 0]

        s, pv = KS(n_eff, n_model)

        if self.verb:
            if pv > alpha:
                print(
                    "We cannot reject the null hypothesis: the empirical and theoretical\
                    distributions has to be considered equivalent"
                )
            else:
                print(
                    "We have to reject the null hypothesis: the two distributions are not\
                    equivalent and thus the model as it is cannot be used to infer the id"
                )

        if pdf:
            fileout = filename + "_mv_pdf.pdf" if (filename is not None) else None
            ddp.plot_pdf(n_eff, n_model, title, fileout)

        if cdf:
            fileout = filename + "_mv_cdf.pdf" if (filename is not None) else None
            ddp.plot_cdf(n_eff, n_model, title, fileout)

        return s, pv

    # ----------------------------------------------------------------------------------------------

    def set_id(self, d):
        """Set id manually.

        Args:
            d (float): intrinsic dimension to assign to the class

        """
        assert d > 0, "cannot support negative dimensions (yet)"
        self.intrinsic_dim = d

    # ----------------------------------------------------------------------------------------------

    def set_lk_ln(self, lk, ln):
        """Assign lk and ln to class.

        Args:
            lk (int): radius of external shells
            ln (int): radius of internal shells

        """
        assert (
            isinstance(ln, (np.int8, np.int16, np.int32, np.int64, int)) and ln >= 0
        ), "select a proper integer ln>=0"
        assert (
            isinstance(lk, (np.int8, np.int16, np.int32, np.int64, int)) and lk > 0
        ), "select a proper integer lk>0"
        assert lk > ln, "select lk and ln, s.t. lk > ln"
        self.ln = ln
        self.lk = lk

    # ----------------------------------------------------------------------------------------------

    def set_ratio(self, r):
        """Set ratio between internal and external shells.

        Args:
            r (float): ratio, 0<r<1 is required
        """
        assert isinstance(r, float) and 0 <= r < 1, "select a proper ratio 0<r<1"
        self.ratio = r

    # ----------------------------------------------------------------------------------------------

    def set_w(self, w):
        """Set weights of points.

        Args:
            w (np.ndarray(int)): multiplicity of points. Needs to have the same length of points

        """
        assert len(w) == self.N and all(
            [wi > 0 and isinstance(wi, (np.int, int))] for wi in w
        ), "load proper integer weights"
        self._weights = np.array(w, dtype=np.int)

    # ----------------------------------------------------------------------------------------------

    def _my_mask(self, subset=None):
        """Compute the mask to select points to be used when computing the id.

        This function create a mask used to compute the id on a subset of datapoints.
        It takes into account mask given by the fix_k or fix_lk methods and integrate
        it with the subset given as an argument. This can be an integer (and points
        will be chosen randomly) or a list/np.ndarray of indices.
        If subset is left empty, just passes the mask from fix_k or fix_lk

        Args:
            subset (int or np.ndarray(int) or list(int), default=None)

        Returns:
            my_mask (np.ndarray(bool)): mask indicating which points will be used in id computation

        """
        assert self._mask is not None

        if subset is None:
            return np.copy(self._mask)

        if isinstance(subset, (np.ndarray, list)):
            assert isinstance(
                subset[0], (int, np.integer)
            ), "elements of list/array must be integers, in order to be used as indexes"
            assert (
                max(subset) < self.N
            ), "the array must contain elements with indexes lower than the total number of elements"

            my_mask = np.zeros(self._mask.shape[0], dtype=bool)
            my_mask[subset] = True
            my_mask *= self._mask  # remove points with bad statistics

        elif isinstance(subset, (int, np.integer)):
            if subset > self._mask.sum():
                my_mask = np.copy(self._mask)

            else:
                my_mask = np.zeros(self._mask.shape[0], dtype=bool)
                idx = np.sort(
                    self.rng.choice(
                        np.where(self._mask)[0],
                        subset,
                        replace=False,
                        shuffle=False,
                    )
                )
                my_mask[idx] = True

                assert my_mask.sum() == subset

        else:
            print("use a proper format for the subset, returning no subset")
            return np.copy(self._mask)

        return my_mask


# ----------------------------------------------------------------------------------------------


if __name__ == "__main__":
    X = rng.integers(0, 20, size=(1000, 2))
    ide = IdDiscrete(coordinates=X)
    ide.compute_distances(maxk=30, metric="manhattan", period=20)
    ide.compute_id_binomial_lk(4, 2)

    print(ide.intrinsic_dim)
