import time

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
def _nrmaxl(floatTYPE_t rinit,
            DTYPE_t kstar_i,
            np.ndarray[floatTYPE_t, ndim = 1] vi):

    # declarations
    cdef DTYPE_t j,niter

    cdef floatTYPE_t a,b,L0,stepmax,ga,gb,jf,t,s,tt,func,sigma,sa,sb

    cdef np.ndarray[floatTYPE_t, ndim = 2] Cov2 = np.zeros((2, 2))
    cdef np.ndarray[floatTYPE_t, ndim = 2] Covinv2 = np.zeros((2, 2))

    cdef floatTYPE_t fepsilon=np.finfo(float).eps


    #useful subroutine
    def _matinv2(np.ndarray[floatTYPE_t, ndim = 2] A, np.ndarray[floatTYPE_t, ndim = 2] B):
    # Performs a direct calculation of the inverse of a 2×2 matrix.
        cdef floatTYPE_t detinv
        #cdef np.ndarray[floatTYPE_t, ndim = 2] B = np.zeros((2, 2))

        detinv = 1./(A[0,0]*A[1,1] - A[0,1]*A[1,0])
        B[0,0] = +detinv * A[1,1]
        B[1,0] = -detinv * A[1,0]
        B[0,1] = -detinv * A[0,1]
        B[1,1] = +detinv * A[0,0]

    # function
    b=rinit
    L0=0.
    a=0.
    stepmax=0.1*abs(b)
    gb=float(kstar_i)
    ga= float(kstar_i + 1) * float(kstar_i) / 2.


    for j in range(kstar_i):
        jf=float(j+1)
        t=b+a*jf
        s=exp(t)
        tt=vi[j]*s
        L0=L0+t-tt
        gb=gb-tt
        ga=ga-jf*tt
        Cov2[0,0]=Cov2[0,0]-tt
        Cov2[0,1]=Cov2[0,1]-jf*tt
        Cov2[1,1]=Cov2[1,1]-jf*jf*tt
    Cov2[1,0]=Cov2[0,1]

    _matinv2(Cov2,Covinv2)
    func=100.
    niter=0

    while ( ((func)>1e-3) and (niter < 10000) ):
        sb=(Covinv2[0,0]*gb+Covinv2[0,1]*ga)
        sa=(Covinv2[1,0]*gb+Covinv2[1,1]*ga)
        niter=niter+1
        sigma=0.1
        if (abs(sigma*sb) > stepmax) :
            sigma=abs(stepmax/sb)
        b=b-sigma*sb
        a=a-sigma*sa

        L0=0.
        gb= float(kstar_i)
        ga= float(kstar_i + 1) * float(kstar_i) / 2.
        Cov2[0,0]=0. #gbb
        Cov2[0,1]=0. #gab
        Cov2[1,1]=0. #gaa
        for j in range(kstar_i):
            jf=float(j+1)
            t=b+a*jf
            s=exp(t)
            tt=vi[j]*s
            L0=L0+t-tt
            gb=gb-tt
            ga=ga-jf*tt
            Cov2[0,0]=Cov2[0,0]-tt
            Cov2[0,1]=Cov2[0,1]-jf*tt
            Cov2[1,1]=Cov2[1,1]-jf*jf*tt
        Cov2[1,0]=Cov2[0,1]
        #Covinv2=np.linalg.inv(Cov2)
        #Covinv2=_matinv2(Cov2)
        _matinv2(Cov2,Covinv2)
        if ((abs(a) <= fepsilon ) or (abs(b) <= fepsilon )):
            func=max(abs(gb),abs(ga))
        else:
            func=max(abs(gb/b),abs(ga/a))

    #Cov2=-Cov2
    #Covinv2=np.linalg.inv(Cov2)
    #Covinv2=_matinv2(Cov2)
    #_matinv2(Cov2,Covinv2)

    return b
