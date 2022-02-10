import numpy as np
import scipy

import dadapy.utils_.utils as ut

# --------------------------------------------------------------------------------------

# bounds for numerical estimation, change if needed
D_MAX = 50.0
D_MIN = np.finfo(np.float32).eps

# TODO: find a proper way to load the data with a relative path
# load, just once and for all, the coefficients for the polynomials in d at fixed L
import os

volumes_path = os.path.join(os.path.split(__file__)[0], "discrete_volumes")
coeff = np.loadtxt(volumes_path + "/L_coefficients_float.dat", dtype=np.float64)

# V_exact_int = np.loadtxt(volume_path + '/V_exact.dat',dtype=np.uint64)

# --------------------------------------------------------------------------------------


def compute_discrete_volume(L, d, O1=False):
    """Enumerate the points contained in a region of radius L according to Manhattan metric

    Args:
            L (nd.array( integer or float )): radii of the volumes of which points will be enumerated
            d (float): dimension of the metric space
            O1 (bool, default=Flase): first order approximation in the large L limit. Set to False in order to have the o(1/L) approx

    Returns:
            V (nd.array( integer or float )): points within the given volumes

    """
    # if L is one dimensional make it an array
    if isinstance(L, (int, np.integer, float, np.float)):
        L = [L]

    # explicit conversion to array of integers
    l = np.array(L, dtype=np.int)

    # exact formula for integer d, cannot be used for floating values
    if isinstance(d, (int, np.integer)):
        V = 0
        for k in range(0, d + 1):
            V += scipy.special.binom(d, k) * scipy.special.binom(l - k + d, d)
        return V

    else:
        # exact enumerating formula for non integer d. Use the loaded coefficients to compute
        # the polynomials in d at fixed (small) L.
        # Exact within numerical precision, as far as the coefficients are available
        def V_polynomials(ll):
            D = d ** np.arange(coeff.shape[1], dtype=np.double)
            V_poly = np.dot(coeff, D)
            return V_poly[ll]

        # Large L approximation obtained using Stirling formula
        def V_Stirling(ll):
            if O1:
                correction = 2**d
            else:
                correction = (
                    np.exp(0.5 * (d + d**2) / ll) * (1 + np.exp(-d / ll)) ** d
                )

            return ll**d / scipy.special.factorial(d) * correction

        ind_small_l = l < coeff.shape[0]
        V = np.zeros(l.shape[0])
        V[ind_small_l] = V_polynomials(l[ind_small_l])
        V[~ind_small_l] = V_Stirling(l[~ind_small_l])

        return V


# --------------------------------------------------------------------------------------


def compute_derivative_discrete_vol(l, d):
    """compute derivative of discrete volumes with respect to dimension

    Args:
            L (int): radii at which the derivative is calculated
            d (float): embedding dimension

    Returns:
            dV_dd (ndarray(float) or float): derivative at different values of radius

    """

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


def compute_jacobian(lk, ln, d):
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
    Vk = compute_discrete_volume(lk, d)[0]
    Vn = compute_discrete_volume(ln, d)[0]
    dVk_dd = compute_derivative_discrete_vol(lk, d)
    dVn_dd = compute_derivative_discrete_vol(ln, d)
    dp_dd = dVn_dd / Vk - dVk_dd * Vn / Vk / Vk
    return dp_dd


# --------------------------------------------------------------------------------------


def compute_binomial_logL(d, Rk, k, Rn, n, discrete=True, w=1):
    """Compute the binomial log likelihood given Rk,Rn,k,n

    Args:
            d (float): embedding dimension
            Rk (ndarray(float) or float): external radii
            k (ndarray(int) or int): number of points within the external radii
            Rn (ndarray(float) or float): external radii
            n (ndarray(int)): number of points within the internal radii
            discrete (bool, default=False): choose discrete or continuous volumes formulation
            w (ndarray(int or float), default=1): weights or multeplicity for each point

    Returns:
            -LogL (float): minus total likelihood

    """

    if discrete:

        p = compute_discrete_volume(Rn, d) / compute_discrete_volume(Rk, d)

    else:
        p = (Rn / Rk) ** d

    if np.any(p == 0):
        print("something went wrong in the calculation of p: check radii and d used")

    # the binomial coefficient is present within the definition of the likelihood,\
    # however it enters additively into the LogL. As such it does not modify its shape.\
    # Neglected if we need only to maximize the LogL

    # log_binom = np.log(scipy.special.binom(k, n))

    # for big values of k and n (~1000) the binomial coefficients explode -> use
    # its logarithmic definition through Stirling apporximation

    # if np.any(log_binom == np.inf):
    #     mask = np.where(log_binom == np.inf)[0]
    #     log_binom[mask] = ut.log_binom_stirling(k[mask], n[mask])

    LogL = n * np.log(p) + (k - n) * np.log(1.0 - p)  # + log_binom
    # add weights cotribution
    LogL = LogL * w
    # returns -LogL in order to be able to minimise it through scipy
    return -LogL.sum()


# --------------------------------------------------------------------------------------


def Cramer_Rao(d, ln, lk, N, k):
    """Calculate the Cramer Rao lower bound for the variance associated with the binomial estimator

    Args:
        d (float): space dimension
        ln (int): radius of the external shell
        lk (int): radius of the internal shell
        N (int): number of points of the dataset
        k (float): average number of neighbours in the external shell

    """

    P = compute_discrete_volume(ln, d)[0] / compute_discrete_volume(lk, d)[0]

    return P * (1 - P) / (np.float64(N) * compute_jacobian(lk, ln, d) ** 2 * k)


# --------------------------------------------------------------------------------------


def eq_to_find0(d, ln, lk, n, k):
    return compute_discrete_volume(ln, d) / compute_discrete_volume(lk, d) - n / k


# --------------------------------------------------------------------------------------


def find_d(ln, lk, n, k):
    if (
        n < 0.00001
    ):  # i.e. i'm dealing with an isolated point, there's no statistics on n
        return 0
    #    if abs(k-n)<0.00001: #i.e. there's internal and external shell have the same amount of points
    #        return 0
    return scipy.optimize.root_scalar(
        eq_to_find0, args=(ln, lk, n, k), bracket=(D_MIN, D_MAX)
    ).root


# --------------------------------------------------------------------------------------


def _beta_prior_d(k, n, lk, ln, a0=1, b0=1, plot=True, verbose=True):
    """Compute the posterior distribution of d given the input aggregates
    Since the likelihood is given by a binomial distribution, its conjugate prior is a beta distribution.
    However, the binomial is defined on the ratio of volumes and so do the beta distribution. As a
    consequence one has to change variable to have the distribution over d

    Args:
            k (nd.array(int)): number of points within the external shells
            n (nd.array(int)): number of points within the internal shells
            lk (int): outer shell radius
            lk (int): inner shell radius
            a0 (float): beta distribution parameter, default =1 for flat prior
            b0 (float): prior initializer, default =1 for flat prior
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
        p = compute_discrete_volume(ln, d) / compute_discrete_volume(lk, d)
        dp_dd = compute_jacobian(lk, ln, d)
        return abs(posterior.pdf(p) * dp_dd)

    # in principle we don't know where the distribution is peaked, so
    # we perform a blind search over the domain
    dx = 1.0
    d_left = D_MIN
    d_right = D_MAX + dx + d_left
    d_range = np.arange(d_left, d_right, dx)
    P = np.array([p_d(di) for di in d_range]) * dx
    counter = 0
    mask = P != 0
    elements = mask.sum()
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
    dx = d_range[1] - d_range[0]
    P = np.array([p_d(di) for di in d_range]) * dx
    P = P.reshape(P.shape[0])

    #    if verbose:
    #        print("iter no\t", counter,'\nd_left\t', d_left,'\nd_right\t', d_right, elements)

    if plot:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(d_range, P)
        plt.xlabel("d")
        plt.ylabel("P(d)")
        plt.title("posterior of d")
        plt.show()

    E_d_emp = np.dot(d_range, P)
    S_d_emp = np.sqrt((d_range * d_range * P).sum() - E_d_emp * E_d_emp)
    if plot:
        print("empirical average:\t", E_d_emp, "\nempirical std:\t\t", S_d_emp)
    #   theoretical results, valid only in the continuum case
    #   E_d = ( sp.digamma(a) - sp.digamma(a+b) )/np.log(r)
    #   S_d = np.sqrt( ( sp.polygamma(1,a) - sp.polygamma(1,a+b) )/np.log(r)**2 )

    return E_d_emp, S_d_emp, d_range, P


# --------------------------------------------------------------------------------------
