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

import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib import cm
from scipy.optimize import minimize_scalar as SMin
from scipy.special import binom, gamma, hyp2f1

from dadapy._cython.cython_distances import _return_hamming_condensed as hc
from dadapy._cython.cython_distances import _return_hamming_condensed_parallel as hcp
from dadapy._cython.cython_distances import _return_manhattan_condensed as mc
from dadapy._cython.cython_distances import _return_manhattan_condensed_parallel as mcp
from dadapy._utils import get_data

# --------------------------------------------------------------------------------------
# bounds for numerical estimation, change if needed
D_MAX = 70.0
D_MIN = np.finfo(np.float32).eps

# load, just once and for all, the coefficients for the polynomials in d at fixed L

# volumes_path = os.path.join(os.path.split(__file__)[0], "discrete_volumes")
# coeff = np.loadtxt(volumes_path + "/L_coefficients_float.dat", dtype=np.float64)

coeff = np.loadtxt(get_data("L_coefficients_float.dat"), dtype=np.float64)
# V_exact_int = np.loadtxt(get_data('V_exact.dat'),dtype=np.uint64)

# --------------------------------------------------------------------------------------


def compute_discrete_volume(l, d, O1=False):
    """Enumerate the points contained in a region of radius L according to Manhattan metric

    Args:
        l (nd.array( integer or float )): radii of the volumes of which points will be enumerated
        d (float): dimension of the metric space

    Returns:
        V (nd.array( integer or float )): points within the given volumes

    """

    # OLD DEFINITIONS using series expansion with eventual approximation for large L

    # O1 (bool, default=False): first order approximation in the large L limit. Set to False in order to have the o(1/L) approx
    # if L is one dimensional make it an array
    # if isinstance(L, (int, np.integer, float, np.float)):
    #     l = [l]

    # # explicit conversion to array of integers
    # l = np.array(L, dtype=np.int)

    # # exact formula for integer d, cannot be used for floating values
    # if isinstance(d, (int, np.integer)):
    #     V = 0
    #     for k in range(0, d + 1):
    #         V += scipy.special.binom(d, k) * scipy.special.binom(l - k + d, d)
    #     return V

    # else:
    #     # exact enumerating formula for non integer d. Use the loaded coefficients to compute
    #     # the polynomials in d at fixed (small) L.
    #     # Exact within numerical precision, as far as the coefficients are available
    #     def V_polynomials(ll):
    #         D = d ** np.arange(coeff.shape[1], dtype=np.double)
    #         V_poly = np.dot(coeff, D)
    #         return V_poly[ll]

    #     # Large L approximation obtained using Stirling formula
    #     def V_Stirling(ll):
    #         if O1:
    #             correction = 2**d
    #         else:
    #             correction = (
    #                 np.exp(0.5 * (d + d**2) / ll) * (1 + np.exp(-d / ll)) ** d
    #             )

    #         return ll**d / scipy.special.factorial(d) * correction

    #     ind_small_l = l < coeff.shape[0]
    #     V = np.zeros(l.shape[0])
    #     V[ind_small_l] = V_polynomials(l[ind_small_l])
    #     V[~ind_small_l] = V_Stirling(l[~ind_small_l])

    #     return V

    # EXACT DEFINITION, acknowledgments to Mathematica

    return binom(d + l, d) * hyp2f1(-d, -l, -d - l, -1)


# --------------------------------------------------------------------------------------


def _compute_derivative_discrete_vol(l, d):
    """compute derivative of discrete volumes with respect to dimension

    Args:
        l (int): radii at which the derivative is calculated
        d (float): embedding dimension

    Returns:
        dV_dd (ndarray(float) or float): derivative at different values of radius

    """

    # TODO: write derivative of expression above. Might do it numerically, as
    #  analytically it is not that simple. For the time being, we use the exact polynomial formulation

    # exact formula with polynomials, for small L
    #    assert isinstance(l, (int, np.int))

    if l < coeff.shape[0]:
        l = int(l)

        D = d ** np.arange(-1, coeff.shape[1] - 1, dtype=np.double)

        coeff_d = coeff[l] * np.arange(
            coeff.shape[1]
        )  # usual coefficient * coeff from first deriv
        return np.dot(coeff_d, D)

    # faster version in case of array l, use 'if all(l<coeff.shape[0])'
    # else:
    # 	L = np.array(L, dtype=np.int)
    # 	coeff_d = coeff*np.arange(coeff.shape[1])
    # 	dV_dd = np.dot(coeff_d, D)
    # 	return dV_dd[L]

    # approximate definition for large L
    else:
        return (
            np.e ** (((0.5 + 0.5 * d) * d) / l)
            * (1 + np.e ** (-d / l)) ** d
            * l**d
            * (
                scipy.special.factorial(d)
                * (
                    (0.5 + d) / l
                    - d / (l + np.e ** (d / l) * l)
                    + np.log(1.0 + np.e ** (-(d / l)))
                    + np.log(l)
                )
                - d * scipy.special.gamma(d) * scipy.special.digamma(1 + d)
            )
        ) / scipy.special.factorial(d) ** 2


# --------------------------------------------------------------------------------------


def _compute_jacobian(lk, ln, d):
    """Compute jacobian of the ratio of volumes wrt d

    Given that the probability of the binomial process is p = V(ln,d)/V(lk,d), in order to
    obtain relationships for d (like deriving the LogLikelihood or computing the posterior)
    one needs to compute the differential dp/dd

    Args:
        lk (int): radius of external volume
        ln (int): radius of internal volume
        d (float): embedding dimension

    Returns:
        dp_dd (ndarray(float) or float): differential

    """
    # p = Vn / Vk
    Vk = compute_discrete_volume(lk, d)  # [0]
    Vn = compute_discrete_volume(ln, d)  # [0]
    dVk_dd = _compute_derivative_discrete_vol(lk, d)
    dVn_dd = _compute_derivative_discrete_vol(ln, d)
    dp_dd = dVn_dd / Vk - dVk_dd * Vn / Vk / Vk
    return dp_dd


# --------------------------------------------------------------------------------------


def _compute_binomial_logl(d, Rk, k, Rn, n, w=1):
    """Compute the binomial log likelihood given Rk,Rn,k,n

    Args:
        d (float): embedding dimension
        Rk (np.ndarray(float) or float): external radii
        k (np.ndarray(int) or int): number of points within the external radii
        Rn (np.ndarray(float) or float): external radii
        n (np.ndarray(int)): number of points within the internal radii
        w (np.ndarray(int or float), default=1): weights or multiplicity for each point

    Returns:
        -LogL (float): minus total likelihood

    """

    p = compute_discrete_volume(Rn, d) / compute_discrete_volume(Rk, d)

    if np.any(p == 0):
        print("something went wrong in the calculation of p: check radii and d used")

    # the binomial coefficient is present within the definition of the likelihood,\
    # however it enters additively into the LogL. As such it does not modify its shape.\
    # Neglected if we need only to maximize the LogL

    # log_binom = np.log(scipy.special.binom(k, n))

    # for big values of k and n (~1000) the binomial coefficients explode -> use
    # its logarithmic definition through Stirling approximation

    # if np.any(log_binom == np.inf):
    #     mask = np.where(log_binom == np.inf)[0]
    #     log_binom[mask] = ut.log_binom_stirling(k[mask], n[mask])

    LogL = n * np.log(p) + (k - n) * np.log(1.0 - p)  # + log_binom
    # add weights contribution
    LogL = LogL * w
    # returns -LogL in order to be able to minimise it through scipy
    return -LogL.sum()


# --------------------------------------------------------------------------------------


def binomial_cramer_rao(d, ln, lk, N, k):
    """Calculate the Cramer Rao lower bound for the variance associated with the binomial estimator

    Args:
        d (float): space dimension
        ln (int): radius of the external shell
        lk (int): radius of the internal shell
        N (int): number of points of the dataset
        k (float): average number of neighbours in the external shell

    Returns:
        cr (float): the Cramer-Rao estimation
    """

    p = compute_discrete_volume(ln, d) / compute_discrete_volume(lk, d)

    return p * (1 - p) / (np.float64(N) * _compute_jacobian(lk, ln, d) ** 2 * k)


# --------------------------------------------------------------------------------------


def _eq_to_find_0(d, ln, lk, n, k):
    """Equation whose root gives the id when fixing radii
    Args:
        d (float): id, parameter to infer
        ln (int): internal radius
        lk (int): external radius
        n (float): average number of points within internal shells
        k (float): average number of points within external shells
    Returns:
    """
    return compute_discrete_volume(ln, d) / compute_discrete_volume(lk, d) - n / k


# --------------------------------------------------------------------------------------


def find_d_root(ln, lk, n, k):
    """Find root (i.e. the intrinsic dimension) of polynomials
    Args:
        ln (int): internal radius
        lk (int): external radius
        n (float): average number of points within internal shells
        k (float): average number of points within external shells
    Returns:
        id (float): the estimated intrinsic dimension
    """
    if (
        n < 0.00001
    ):  # i.e. i'm dealing with a isolated points, there's no statistics on n
        return 0
    #    if abs(k-n)<0.00001: #i.e. there's internal and external shell have the same amount of points
    #        return 0
    return scipy.optimize.root_scalar(
        _eq_to_find_0,
        args=(ln, lk, n, k),
        bracket=(D_MIN + np.finfo(np.float16).eps, D_MAX),
    ).root


# --------------------------------------------------------------------------------------


def find_d_likelihood(ln, lk, n, k, ww):
    """Finds the ID as maximum of the likelihood
    Args:
        ln (int or np.ndarray(int)): internal radius
        lk (int or np.ndarray(int)): external radius
        n (np.ndarray(int)): number of points within internal shells
        k (np.ndarray(int)): number of points within external shells
        ww (np.ndarray(int)): multiplicity of points
    Returns:
        id (float): the estimated intrinsic dimension
    """

    return SMin(
        _compute_binomial_logl,
        args=(lk, k, ln, n, ww),
        bounds=(D_MIN + np.finfo(np.float16).eps, D_MAX),
        method="bounded",
    ).x


# --------------------------------------------------------------------------------------


def profile_likelihood(ln, lk, n, k, ww, plot=False):
    """Compute the likelihood of the binomial process and find the ID when ln and lk vary for each point

    Args:
        ln (np.ndarray(int)): inner shell radii
        lk (np.ndarray(int)): outer shell radii
        n (np.ndarray(int)): number of points within the internal shells
        k (np.ndarray(int)): number of points within the external shells
        ww (np.ndarray(int)): multiplicity of datapoints
        plot (bool, default=False): plot the posterior
    Returns:
        E_d_emp (float): mean value of the posterior
        S_d_emp (float): std of the posterior
        d_range (ndarray(float)): domain of the posterior
        P (ndarray(float)): probability of the posterior
    """

    def p_d(d):
        return _compute_binomial_logl(d, lk, k, ln, n, w=ww)

    # in principle, we don't know where the distribution is peaked, so
    # we perform a blind search over the domain
    dx = 10.0
    d_left = D_MIN
    d_right = D_MAX + dx + d_left
    elements = 0
    counter = 0
    # if less than 3 points have P!=0 are found, reduce the interval
    while elements < 3:
        dx /= 10.0
        counter += 1
        #        print('iter no.', counter, '\nmesh size', dx)
        d_range = np.arange(d_left, d_right, dx)
        P = np.array([p_d(di) for di in d_range])  # * dx
        P = P.reshape(P.shape[0])
        P -= P.min()
        P = np.exp(-P)
        mask = P > 1e-20
        elements = mask.sum()

    # with more than 3 points where P!=0 we can restrict the domain in that interval and build a smooth distribution
    # I choose 1000 points but such quantity can be varied according to necessity
    ind = np.where(mask)[0]
    d_left = d_range[ind[0]] - 0.5 * dx if d_range[ind[0]] - dx > 0 else D_MIN
    d_right = d_range[ind[-1]] + 0.5 * dx
    d_range = np.linspace(d_left, d_right, 1000)
    dx = d_range[1] - d_range[0]
    P = np.array([p_d(di) for di in d_range])  # * dx
    P = P.reshape(P.shape[0])
    P -= P.min()
    P = np.exp(-P)
    P /= P.sum()
    # if verbose:
    #   print("iter no\t", counter,'\nd_left\t', d_left,'\nd_right\t', d_right, elements)

    if plot:
        plt.figure()
        plt.plot(d_range, P)
        plt.xlabel("d")
        plt.ylabel("P(d)")
        plt.title("Posterior of d")

    E_d_emp = np.dot(d_range, P)
    S_d_emp = np.sqrt((d_range * d_range * P).sum() - E_d_emp * E_d_emp)
    if plot:
        print("empirical average:\t", E_d_emp, "\nempirical std:\t\t", S_d_emp)

    return E_d_emp, S_d_emp, d_range, P


# --------------------------------------------------------------------------------------
def beta_prior_d(k, n, lk, ln, a0=1, b0=1, plot=True):
    """Compute the posterior distribution of d given the input aggregates
    Since the likelihood is given by a binomial distribution, its conjugate prior is a beta distribution.
    However, the binomial is defined on the ratio of volumes and so do the beta distribution. As a
    consequence one has to change variable to have the distribution over d

    Args:
        k (nd.array(int)): number of points within the external shells
        n (nd.array(int)): number of points within the internal shells
        lk (int): outer shell radius
        ln (int): inner shell radius
        a0 (float): beta distribution parameter, default =1 for flat prior
        b0 (float): prior initializer, default =1 for flat prior
        plot (bool, default=False): plot the posterior
    Returns:
        E_d_emp (float): mean value of the posterior
        S_d_emp (float): std of the posterior
        d_range (ndarray(float)): domain of the posterior
        P (ndarray(float)): probability of the posterior
    """
    # from scipy.special import beta as beta_f
    from scipy.stats import beta as beta_d

    a = a0 + n.sum()
    b = b0 + k.sum() - n.sum()
    posterior = beta_d(a, b)

    def p_d(d):
        p = compute_discrete_volume(ln, d) / compute_discrete_volume(lk, d)
        dp_dd = _compute_jacobian(lk, ln, d)
        return abs(posterior.pdf(p) * dp_dd)

    dx = 10.0
    d_left = D_MIN
    d_right = D_MAX + dx + d_left
    counter = 0
    elements = 0
    # in principle, we don't know where the distribution is peaked, so
    # we perform a blind search over the domain
    # if less than 3 points !=0 are found, reduce the interval
    #    print('building posterior')
    while elements < 3:
        dx /= 10
        counter += 1
        #        print('inter no.', counter, '\nmesh size:', dx)
        d_range = np.arange(d_left, d_right, dx)
        P = np.array([p_d(di) for di in d_range])  # * dx
        mask = P > 1e-20
        elements = mask.sum()

    # with more than 3 points where P!=0, we can restrict the domain in that interval and build a smooth distribution
    # I choose 1000 points but such quantity can be varied according to necessity
    ind = np.where(mask)[0]
    d_left = d_range[ind[0]] - 0.5 * dx if d_range[ind[0]] - dx > 0 else D_MIN
    d_right = d_range[ind[-1]] + 0.5 * dx
    d_range = np.linspace(d_left, d_right, 1000)
    dx = d_range[1] - d_range[0]
    P = np.array([p_d(di) for di in d_range])  # * dx
    P = P.reshape(P.shape[0])
    P /= P.sum()
    #    if verbose:
    #        print("iter no\t", counter,'\nd_left\t', d_left,'\nd_right\t', d_right, elements)

    if plot:
        plt.figure()
        plt.plot(d_range, P)
        plt.xlabel("d")
        plt.ylabel("P(d)")
        plt.title("posterior of d")

    E_d_emp = np.dot(d_range, P)
    S_d_emp = np.sqrt((d_range * d_range * P).sum() - E_d_emp * E_d_emp)
    if plot:
        print("empirical average:\t", E_d_emp, "\nempirical std:\t\t", S_d_emp)

    return E_d_emp, S_d_emp, d_range, P


# --------------------------------------------------------------------------------------
# HELPER FUNCTIONS TO COMPUTE DISTANCES IN THE CONDENSED FORM


def return_condensed_distances(points, metric, d_max=100, period=None, n_jobs=1):
    """Compute number of points at each distance
    
    Instead of focusing on the distance of each neighbour, it just saves how many neighbours are
    present at each distance and returns their cumulative distribution.
    This saves loads of memory, as instead of having a matrix NxN (or Nxk_max)
    it just saves N x d_max, where generally d_max is of order of 100, while k_max 
    needs to be a finite fraction of N to have reliable results.
    If you want to know the neighbours, compute distances with condensed=False

    Args:
        points (np.ndarray(int/strings)): datapoints
        metric (string): metric used to compute distances, 'manhattan' and 'hamming' supported so far
        d_max (int, default=100): max distance around each point to look at
        period (optional, float or np.ndarray(float)): possible PBC boundaries
        n_jobs (int): number of CPUs used to make the calculation
    Returns:
        distances (np.ndarray(int,int)): N x d_max matrix of cumulatives number of points at \
                                     successive distances
        indices (optional, np.ndarray(int,int)): N x maxk_ind matrix of neighbours indices
    """

    if metric == "manhattan":
        return manhattan_distances_condensed(points, d_max, period, n_jobs)
    elif metric == "hamming":
        return hamming_distances_condensed(points, d_max, n_jobs)
    else:
        print(
            'insert a proper metric: up to now the supported ones are "manhattan" and "hamming"'
        )
        return 0


# --------------------------------------------------------------------------------------


def manhattan_distances_condensed(points, d_max=100, period=None, n_jobs=1):
    """Compute condensed distances according to manhattan metric
    Args:
        points (np.ndarray(int)): datapoints
        d_max (int, default=100): max distance around each point to look at
        period (float or np.ndarray(float)): PBC boundaries
        n_jobs (int): number of CPUs used to make the calculation
    Returns:
        distances (np.ndarray(int,int)): N x d_max matrix of cumulatives number of points at \
                                     successive distances
    """

    d_max = min(d_max, points.shape[1] * points.max())
    if n_jobs > 2:
        return mcp(points, d_max, period, n_jobs), None
    else:
        return mc(points, d_max, period), None


# --------------------------------------------------------------------------------------


def manhattan_distances_idx(points, d_max=100, maxk_ind=None, period=None):
    """Compute condensed distances according to manhattan metric

    Python version of the function above, slower. Possibly returns indices of NN

    Args:
        points (np.ndarray(int/strings)): datapoints
        d_max (int, default=100): max distance around each point to look at
        maxk_ind (int, default=None): number of neighbours' indices to be stored
        period (float or np.ndarray(float)): PBC boundaries
    Returns:
        distances (np.ndarray(int,int)): N x d_max matrix of cumulatives number of points at \
                                     successive distances
        indices (optional, np.ndarray(int,int)): N x maxk_ind matrix of neighbours indices
    """

    dist_count = np.zeros((points.shape[0], d_max + 1), dtype=int)

    if maxk_ind is not None:
        indexes = np.zeros((points.shape[0], maxk_ind), dtype=int)

    for i, pt in enumerate(points):
        if period is None:
            appo = np.sum(abs(pt - points), axis=1, dtype=int)
        else:
            appo = pt - points
            appo = np.sum(
                abs(appo - np.rint(appo / period) * period), axis=1, dtype=int
            )

        if maxk_ind is None:
            uniq, counts = np.unique(appo, return_counts=True)
        else:
            index_i = np.argsort(appo)
            indexes[i] = np.copy(index_i[:maxk_ind])
            uniq, counts = np.unique(appo[index_i], return_counts=True)

        assert uniq[-1] <= d_max
        dist_count[i, uniq] = np.copy(counts)
        dist_count[i] = np.cumsum(dist_count[i])

    if maxk_ind is None:
        return dist_count, None
    else:
        return dist_count, indexes


# --------------------------------------------------------------------------------------


def hamming_distances_condensed(points, d_max, n_jobs=1):
    """Compute condensed distances according to hamming metric
    Args:
        points (np.ndarray(int/strings)): datapoints
        d_max (int, default=100): max distance around each point to look at
        n_jobs (int): number of CPUs used to make the calculation
    Returns:
        distances (np.ndarray(int,int)): N x d_max matrix of cumulatives number of points at \
                                     successive distances
    """
    d_max = min(d_max, points.shape[1])
    if n_jobs > 2:
        return hcp(points, d_max, n_jobs), None
    else:
        return hc(points, d_max), None


# --------------------------------------------------------------------------------------


def hamming_distances_idx(points, d_max=100, maxk_ind=None):
    """Compute condensed distances according to hamming metric

    Python version of the cython function above (slower). Possibly returns the indices of NN if maxk is provided
    Args:
        points (np.ndarray(int/strings)): datapoints
        d_max (int, default=100): max distance around each point to look at
        maxk_ind (int, default=None): number of neighbours' indices to be stored
    Returns:
        distances (np.ndarray(int,int)): N x d_max matrix of cumulatives number of points at \
                                     successive distances
        indices (optional, np.ndarray(int,int)): N x maxk_ind matrix of neighbours indices
    """

    dist_count = np.zeros((points.shape[0], d_max + 1), dtype=int)

    if maxk_ind is not None:
        indexes = np.zeros((points.shape[0], maxk_ind), dtype=int)

    def hamming_distance_couple(a, b):
        assert len(a) == len(b)
        return sum([a[i] != b[i] for i in range(len(a))])

    for i, pt in enumerate(points):
        appo = np.array([hamming_distance_couple(pt, pt_i) for pt_i in points])

        if maxk_ind is None:
            uniq, counts = np.unique(appo, return_counts=True)
        else:
            index_i = np.argsort(appo)
            indexes[i] = np.copy(index_i[:maxk_ind])
            uniq, counts = np.unique(appo[index_i], return_counts=True)

        assert uniq[-1] <= d_max
        dist_count[i, uniq] = np.copy(counts)
        dist_count[i] = np.cumsum(dist_count[i])

    if maxk_ind is None:
        return dist_count, None
    else:
        return dist_count, indexes
