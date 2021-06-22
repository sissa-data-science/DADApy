import numpy as np
import pytest

from duly.id_discrete import *
from duly.utils_.utils import compute_NN_PBC as PBC


def test_id_discrete():
    """Test the discrete id estimator"""

    N = 500

    box = 20

    d = 2

    X = rng.integers(0, box, size=(N, d))

    dist, ind = PBC(X, N - 1, box_size=box, p=1)

    IDD = IdDiscrete(X, maxk=X.shape[0])
    IDD.distances = dist
    IDD.dist_indices = ind

    IDD.compute_id_binomial_k(25, False, 0.5)

    assert pytest.approx(d, IDD.id_estimated_binom)

    IDD.compute_id_binomial_k(4, True, 0.5)

    assert pytest.approx(d, IDD.id_estimated_binom)

    IDD.compute_id_binomial_Lk(4, 2, "mle")

    assert pytest.approx(d, IDD.id_estimated_binom)


from duly.utils_ import discrete_functions as df

"""
def test_id_volumes():

	L_max,d_max = df.coeff.shape[0],21
	V_computed = np.zeros((L_max,d_max))
	for d in range(0,d_max):
    	V_computed[:,d] = np.dot( df.coeff,d**np.arange(0,L_max,dtype=np.double) ) 
    diff = np.array((df.V_exact_int-V_computed),dtype=np.int)
    assert diff.sum()==0
"""
