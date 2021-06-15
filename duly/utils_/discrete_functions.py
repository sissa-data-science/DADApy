import duly.utils_.utils as ut
import numpy as np
# load, just once and for all, the coefficients for the polynomials in d at fixed L
import pkg_resources
import scipy

# --------------------------------------------------------------------------------------


DATA_PATH = pkg_resources.resource_filename("duly.utils_", "/discrete_volumes/")
coeff = np.loadtxt(DATA_PATH + "L_coefficients_float.dat", dtype=np.float64)

# V_exact_int = np.loadtxt(DATA_PATH + 'V_exact.dat',dtype=np.uint64)

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
                correction = 2 ** d
            else:
                correction = (
                    np.exp(0.5 * (d + d ** 2) / ll) * (1 + np.exp(-d / ll)) ** d
                )

            return ll ** d / scipy.special.factorial(d) * correction

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
    assert isinstance(l, (int, np.int))
    if l < coeff.shape[0]:

        D = d ** np.arange(-1, coeff.shape[1] - 1, dtype=np.double)

        coeff_d = coeff[l] * np.arange(coeff.shape[1])
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
            * l ** d
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

    if np.any(p==0) :
    	print('something went wrong in the calculation of p: check radii and d used')

    log_binom = np.log(scipy.special.binom(k, n))

    # for big values of k and n (~1000) the binomial coefficients explode -> use
    # its logarithmic definition through Stirling apporximation
    if np.any(log_binom == np.inf):
        mask = np.where(log_binom == np.inf)[0]
        log_binom[mask] = ut.log_binom_stirling(k[mask], n[mask])

    LogL = log_binom + n * np.log(p) + (k - n) * np.log(1.0 - p)
    # add weights cotribution
    LogL = LogL * w
    # returns -LogL in order to be able to minimise it through scipy
    return -LogL.sum()
