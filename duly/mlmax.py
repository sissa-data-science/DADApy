import autograd.numpy as anp
import numpy as np
# TODO: compute gradient of ML to speedup optimization!
from autograd import grad
from scipy.optimize import minimize

import adpy.utils as ut


def ML_fun_kNN_corr(Fis, args):
    kopts = args[0]
    Vis = args[1]
    dist_indices = args[2]
    Fij_list = args[3]
    Fij_var_list = args[4]
    alpha = args[5]

    L = 0.

    for i, (Fijs, Fijs_var) in enumerate(zip(Fij_list, Fij_var_list)):
        Fi = Fis[i]
        k = kopts[i]
        Li = k * Fi - Vis[i] * anp.exp(Fi)

        for nneigh in range(k):
            j = dist_indices[i, nneigh + 1]

            Li -= alpha * ((Fis[j] - Fi) - Fijs[nneigh]) ** 2 / (2 * Fijs_var[nneigh])

        L += Li

    return - L


ML_fun_grad = grad(ML_fun_kNN_corr)


def ML_fun_gPAk(params, args):
    '''
    The function returns the log-Likelihood expression to be minimized.

    Requirements:

    * **params**: array of initial values for ''a'', ''b''
    * **args**: additional parameters ''kopt'', ''Vi'' entering the Likelihood

    Note:

    * **b**: correspond to the ''log(rho)'', as in Eq. (S1)
    * **a**: the linear correction, as in Eq. (S1)

    '''

    Fi = params[0]

    a = params[1]

    kopt = args[0]

    vij = args[1]

    grads_ij = args[2]

    gb = kopt

    ga = np.sum(grads_ij)

    L0 = Fi * gb + a * ga

    for j in range(kopt):
        t = Fi + a * grads_ij[j]

        s = np.exp(t)

        tt = vij[j] * s

        L0 = L0 - tt

    return -L0


def ML_fun_gpPAk(params, args):
    '''
    The function returns the log-Likelihood expression to be minimized.

    Requirements:

    * **params**: array of initial values for ''a'', ''b''
    * **args**: additional parameters ''kopt'', ''Vi'' entering the Likelihood

    Note:

    * **b**: correspond to the ''log(rho)'', as in Eq. (S1)
    * **a**: the linear correction, as in Eq. (S1)

    '''

    Fi = params[0]

    a = params[1]

    kopt = args[0]

    vij = args[1]

    grads_ij = args[2]

    gb = kopt

    ga = (kopt + 1) * kopt * 0.5

    L0 = Fi * gb + np.sum(grads_ij) + a * ga

    for j in range(kopt):
        jf = float(j + 1)
        t = Fi + grads_ij[j] + a * jf

        s = np.exp(t)

        tt = vij[j] * s

        L0 = L0 - tt

    return -L0


def ML_fun(params, args):
    '''
    The function returns the log-Likelihood expression to be minimized.

    Requirements:

    * **params**: array of initial values for ''a'', ''b''
    * **args**: additional parameters ''kopt'', ''Vi'' entering the Likelihood

    Note:

    * **b**: correspond to the ''log(rho)'', as in Eq. (S1)
    * **a**: the linear correction, as in Eq. (S1)

    '''
    # g = [0, 0]
    b = params[0]
    a = params[1]
    kopt = args[0]
    gb = kopt
    ga = (kopt + 1) * kopt * 0.5
    L0 = b * gb + a * ga
    Vi = args[1]
    for k in range(1, kopt):
        jf = float(k)
        t = b + a * jf
        s = np.exp(t)
        tt = Vi[k - 1] * s
        L0 = L0 - tt
    return -L0


def ML_hess_fun(params, args):
    '''
    The function returns the expressions for the asymptotic variances of the estimated parameters.

    Requirements:

    * **params**: array of initial values for ''a'', ''b''
    * **args**: additional parameters ''kopt'', ''Vi'' entering the Likelihood

    Note:

    * **b**: correspond to the ''log(rho)'', as in Eq. (S1)
    * **a**: the linear correction, as in Eq. (S1)

    '''
    g = [0, 0]
    b = params[0]
    a = params[1]
    kopt = args[0]
    gb = kopt
    ga = (kopt + 1) * kopt * 0.5
    L0 = b * gb + a * ga
    Vi = args[1]
    Cov2 = np.array([[0.] * 2] * 2)
    for k in range(1, kopt):
        jf = float(k)
        t = b + a * jf
        s = np.exp(t)
        tt = Vi[k - 1] * s
        L0 = L0 - tt
        gb = gb - tt
        ga = ga - jf * tt
        Cov2[0][0] = Cov2[0][0] - tt
        Cov2[0][1] = Cov2[0][1] - jf * tt
        Cov2[1][1] = Cov2[1][1] - jf * jf * tt
    Cov2[1][0] = Cov2[0][1]
    Cov2 = Cov2 * (-1)
    Covinv2 = np.linalg.inv(Cov2)

    g[0] = np.sqrt(Covinv2[0][0])
    g[1] = np.sqrt(Covinv2[1][1])
    return g


def MLmax(rr, kopt, Vi):
    '''
    This function uses the scipy.optimize package to minimize the function returned by ''ML_fun'', and
    the ''ML_hess_fun'' for the analytical calculation of the Hessian for errors estimation.
    It returns the value of the density which minimize the log-Likelihood in Eq. (S1)

    Requirements:

    * **rr**: is the initial value for the density, by using the standard k-NN density estimator
    * **kopt**: is the optimal neighborhood size k as return by the Likelihood Ratio test
    * **Vi**: is the list of the ''kopt'' volumes of the shells defined by two successive nearest neighbors of the current point

    # '''
    # results = minimize(ML_fun, [rr, 0.], method='Nelder-Mead', args=([kopt, Vi],),
    #                    options={'maxiter': 1000})

    results = minimize(ML_fun, [rr, 0.], method='Nelder-Mead', tol=1e-6, args=([kopt, Vi]),
                       options={'maxiter': 1000})

    # err = ML_hess_fun(results.x, [kopt, Vi])
    # a_err = err[1]
    rr = results.x[0]  # b
    print(results.message)
    return rr


def MLmax_gPAk(rr, kopt, Vi, grads_ij):
    results = minimize(ML_fun_gPAk, [rr, 0.], method='Nelder-Mead', tol=1e-6,
                       args=([kopt, Vi, grads_ij]),
                       options={'maxiter': 1000})

    rr = results.x[0]  # b
    print(results.message)
    return rr


def MLmax_gpPAk(rr, kopt, Vi, grads_ij):
    results = minimize(ML_fun_gpPAk, [rr, 0.], method='Nelder-Mead', tol=1e-6,
                       args=([kopt, Vi, grads_ij]),
                       options={'maxiter': 1000})

    rr = results.x[0]  # b
    print(results.message)
    return rr


def MLmax_kNN_corr(Fis, kstar, Vis, dist_indices, Fij_list, Fij_var_list, alpha):
    print('ML maximisation started')

    # methods: 'Nelder-Mead', 'BFGS'
    # results = minimize(ML_fun_kNN_corr, Fis, method='Nelder-Mead', tol=1e-6,
    #                    args=([kstar, Vis, dist_indices, Fij_list, Fij_var_list, alpha]),
    #                    options={'maxiter': 50000})

    results = minimize(ML_fun_kNN_corr, Fis, method='CG', tol=1e-6, jac=ML_fun_grad,
                       args=([kstar, Vis, dist_indices, Fij_list, Fij_var_list, alpha]),
                       options={'maxiter': 100})

    rr = results.x  # b
    print(results.message)
    print(results.nit)
    print(results.nfev)
    print(results.njev)
    print(np.mean(abs(results.jac)))
    return rr


### Maximisation of other quantities (not maximum likelihood)

def Symm_Imbalance(params, args):
    X = args[0]
    Xp = args[1]
    maxk = args[2]
    k = args[3]
    ltype = args[4]

    D = X.shape[1]
    #Dp = X.shape[1]

    #Cx = np.hstack([params[:D - 1], 1 ])
    # Cxp = np.hstack([params[D - 1:]**2, 1 - np.sum(params[D - 1:]**2)])

    Cx = params[:D ]#**2
    #Cxp = params[D:]**2

    X = X * Cx[None, :]
    #Xp = Xp * Cxp[None, :]

    lx_xp, l_xp_x = ut._get_loss_between_two(X, Xp, maxk, k, ltype)

    #total_imbalance = (lx_xp + l_xp_x) / 2.

    return lx_xp, l_xp_x


def Min_Symm_Imbalance(X, Xp, maxk, k, ltype, params0):
    results = minimize(Symm_Imbalance, params0, method='Nelder-Mead', tol=1e-5,
                       args=([X, Xp, maxk, k, ltype]), options={'maxiter': 5000})

    print(results.message)
    print(results.nit)
    return results.x, results.fun


if __name__ == '__main__':
    pass
