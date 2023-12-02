
#nthreads = 2

from os import environ
environ["OMP_NUM_THREADS"] = str(nthreads) # export OMP_NUM_THREADS=1
environ["OPENBLAS_NUM_THREADS"] = str(nthreads) # export OPENBLAS_NUM_THREADS=1
environ["MKL_NUM_THREADS"] = str(nthreads) # export MKL_NUM_THREADS=1
environ["VECLIB_MAXIMUM_THREADS"] = str(nthreads) # export VECLIB_MAXIMUM_THREADS=1
environ["NUMEXPR_NUM_THREADS"] = str(nthreads) # export NUMEXPR_NUM_THREADS=1

"""to copy and paste
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
"""



"""to copy and paste
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export VECLIB_MAXIMUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
"""


"""to copy and paste
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
"""