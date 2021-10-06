import pytest

from dadapy.id_discrete import *


def test_id_discrete():
    """Test the discrete id estimator"""

    N = 500

    box = 20

    d = 2

    X = rng.integers(0, box, size=(N, d))

    IDD = IdDiscrete(X, maxk=X.shape[0])
    IDD.compute_distances(p=1, period=box)

    IDD.compute_id_binomial_k(k=25, shell=False, ratio=0.5)

    assert pytest.approx(d, IDD.id_estimated_binom)

    IDD.compute_id_binomial_k(k=4, shell=True, ratio=0.5)

    assert pytest.approx(d, IDD.id_estimated_binom)

    IDD.compute_id_binomial_lk(lk=4, ln=2, method="mle")

    assert pytest.approx(d, IDD.id_estimated_binom)


from dadapy.utils_ import discrete_functions as df

"""
def test_id_volumes():

	L_max,d_max = df.coeff.shape[0],21
	V_computed = np.zeros((L_max,d_max))
	for d in range(0,d_max):
    	V_computed[:,d] = np.dot( df.coeff,d**np.arange(0,L_max,dtype=np.double) ) 
    diff = np.array((df.V_exact_int-V_computed),dtype=np.int)
    assert diff.sum()==0
"""
