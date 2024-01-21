# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import cython
import numpy as np

cimport numpy as np
from libc.math cimport exp

DTYPE = np.int64
floatTYPE = np.float64


ctypedef np.int_t DTYPE_t
ctypedef np.float64_t floatTYPE_t

@cython.boundscheck(False)
@cython.cdivision(True)
def _nrmaxl(np.ndarray[floatTYPE_t, ndim = 1] F,
            np.ndarray[DTYPE_t, ndim = 1] kstar,
            np.ndarray[floatTYPE_t, ndim = 2] volumes):

    # declarations
    cdef DTYPE_t                                i, j, niter, flag, is_singular
    cdef floatTYPE_t                            a, stepmax, lr, grad_a, delta_a, grad_f, delta_f, gf_tmp, l, func, detinv
    cdef floatTYPE_t                            fepsilon    = np.finfo(float).eps
    cdef np.ndarray[floatTYPE_t, ndim = 2]      Hess        = np.zeros((2, 2))
    cdef np.ndarray[floatTYPE_t, ndim = 2]      HessInv     = np.zeros((2, 2))


    N = F.shape[0]
    is_singular = 0
    for i in range(N):

        flag = 0
        for j in range(kstar[i]):
            if volumes[i, j] < 1.0e-300:
                flag = 1
                break

        if flag==0:
            a=0.
            stepmax=0.1*abs(F[i])

            #hessian and gradient update
            grad_f = float(kstar[i])
            grad_a = float(kstar[i] + 1) * float(kstar[i]) / 2.
            Hess[0,0]=0.
            Hess[0,1]=0.
            Hess[1,1]=0.
            for j in range(kstar[i]):
                l=float(j+1)

                gf_tmp = volumes[i, j]*exp(F[i]+a*l)

                grad_f = grad_f - gf_tmp
                grad_a = grad_a - l*gf_tmp

                Hess[0,0] = Hess[0,0] - gf_tmp
                Hess[0,1] = Hess[0,1] - l*gf_tmp
                Hess[1,1] = Hess[1,1] - l**2*gf_tmp
            Hess[1,0] = Hess[0,1]
            detHess = Hess[0,0]*Hess[1,1] - Hess[0,1]*Hess[1,0]
            if detHess< fepsilon:
              is_singular = 1
            else:
              #inversion of the hessian matrix
              detinv = 1./(Hess[0,0]*Hess[1,1] - Hess[0,1]*Hess[1,0])
              HessInv[0,0] = +detinv * Hess[1,1]
              HessInv[1,0] = -detinv * Hess[1,0]
              HessInv[0,1] = -detinv * Hess[0,1]
              HessInv[1,1] = +detinv * Hess[0,0]

            func=100.
            niter=0

            while ( ((func)>1e-3) and (niter < 10000) ):

                if detHess<fepsilon:
                  #parameter update fixed point iteration
                  F[i] = ( -a*kstar[i]*(kstar[i]+1)/2  - (grad_f-kstar[i]) ) / kstar[i]
                  a = (- F[i]*kstar[i] - (grad_f-kstar[i])) / (kstar[i]*(kstar[i]+1)/2)
                else:
                  #parameter step calculation
                  delta_f = (HessInv[0,0]*grad_f+HessInv[0,1]*grad_a)
                  delta_a = (HessInv[1,0]*grad_f+HessInv[1,1]*grad_a)

                  #learning rate/counter update
                  niter=niter+1
                  lr=0.1
                  if (abs(lr*delta_f) > stepmax) :
                      lr=abs(stepmax/delta_f)

                  #parameter update
                  F[i] = F[i] - lr*delta_f
                  a = a - lr*delta_a

                #gradient calculation at F[i]: it must be computed after F update to provide a consistent check of func
                grad_f = float(kstar[i])
                grad_a = float(kstar[i] + 1) * float(kstar[i]) / 2.
                Hess[0,0]=0.
                Hess[0,1]=0.
                Hess[1,1]=0.
                for j in range(kstar[i]):
                    l=float(j+1)

                    gf_tmp = volumes[i, j]*exp(F[i]+a*l)

                    grad_f = grad_f - gf_tmp
                    grad_a = grad_a - l*gf_tmp

                    Hess[0,0] = Hess[0,0] - gf_tmp
                    Hess[0,1] = Hess[0,1] - l*gf_tmp
                    Hess[1,1] = Hess[1,1] - l**2*gf_tmp
                Hess[1,0] = Hess[0,1]

                detHess = Hess[0,0]*Hess[1,1] - Hess[0,1]*Hess[1,0]
                if detHess < fepsilon:
                  is_singular = 1

                else:
                  #inversion of the hessian matrix
                  detinv = 1./(Hess[0,0]*Hess[1,1] - Hess[0,1]*Hess[1,0])
                  HessInv[0,0] = +detinv * Hess[1,1]
                  HessInv[1,0] = -detinv * Hess[1,0]
                  HessInv[0,1] = -detinv * Hess[0,1]
                  HessInv[1,1] = +detinv * Hess[0,0]



                if ((abs(a) <= fepsilon ) or (abs(F[i]) <= fepsilon )):
                    func = max(abs(grad_f),abs(grad_a))
                else:
                    func = max(abs(grad_f/F[i]),abs(grad_a/a))



    return F, is_singular
