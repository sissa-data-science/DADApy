import time

import cython
import numpy as np

cimport numpy as np
from libc.math cimport exp

DTYPE = np.int
floatTYPE = np.float
boolTYPE = np.bool

ctypedef np.int_t DTYPE_t
ctypedef np.float64_t floatTYPE_t

@cython.boundscheck(False)
@cython.cdivision(True)
def _nrmaxl(np.ndarray[floatTYPE_t, ndim = 1] F,
            np.ndarray[DTYPE_t, ndim = 1] kstar,
            np.ndarray[floatTYPE_t, ndim = 2] volumes):

    # declarations
    cdef DTYPE_t                                i, j, niter, flag
    cdef floatTYPE_t                            a, stepmax, lr, grad_a, delta_a, grad_f, delta_f, gf_tmp, l, func, detinv
    cdef floatTYPE_t                            fepsilon    = np.finfo(float).eps
    cdef np.ndarray[floatTYPE_t, ndim = 2]      Hess        = np.zeros((2, 2))
    cdef np.ndarray[floatTYPE_t, ndim = 2]      HessInv     = np.zeros((2, 2))


    #Helper funcion computes the inverse Hessian (2x2)
    # def invert(np.ndarray[floatTYPE_t, ndim = 2] A, np.ndarray[floatTYPE_t, ndim = 2] B):
    #     cdef floatTYPE_t detinv
    #
    #     detinv = 1./(A[0,0]*A[1,1] - A[0,1]*A[1,0])
    #     B[0,0] = +detinv * A[1,1]
    #     B[1,0] = -detinv * A[1,0]
    #     B[0,1] = -detinv * A[0,1]
    #     B[1,1] = +detinv * A[0,0]

    N = F.shape[0]
    for i in range(N):

        flag = 0
        for j in range(kstar[i]):
            if volumes[i, j] < 1.0e-300:
                flag = 1
                break

        if flag==0:
            a=0.
            stepmax=0.1*abs(F[i])
            func=100.
            niter=0

            while ( ((func)>1e-3) and (niter < 10001) ):

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

                #inversion of the hessian matrix
                detinv = 1./(Hess[0,0]*Hess[1,1] - Hess[0,1]*Hess[1,0])
                HessInv[0,0] = +detinv * Hess[1,1]
                HessInv[1,0] = -detinv * Hess[1,0]
                HessInv[0,1] = -detinv * Hess[0,1]
                HessInv[1,1] = +detinv * Hess[0,0]

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

                if ((abs(a) <= fepsilon ) or (abs(F[i]) <= fepsilon )):
                    func = max(abs(grad_f),abs(grad_a))
                else:
                    func = max(abs(grad_f/F[i]),abs(grad_a/a))
    return F
