import numpy as np
import pytest

from dadapy.id_discrete import IdDiscrete

rng = np.random.default_rng(12345)


def test_id_discrete():
    """Test the discrete id estimator"""

    N = 500

    box = 20

    d = 2

    X = rng.integers(0, box, size=(N, d))

    IDD = IdDiscrete(X, maxk=X.shape[0])
    IDD.compute_distances(metric="manhattan", period=box)

    IDD.compute_id_binomial_k(k=25, shell=False, ratio=0.5)

    assert IDD.id_estimated_binom == pytest.approx(2.047335150414252)

    IDD.compute_id_binomial_k(k=4, shell=True, ratio=0.5)

    assert IDD.id_estimated_binom == pytest.approx(2.014968714680048)

    IDD.compute_id_binomial_lk(lk=4, ln=2, method="mle")

    assert IDD.id_estimated_binom == pytest.approx(2.012434143811029)


"""
from dadapy.utils_ import discrete_functions as df

def test_id_volumes():

	L_max,d_max = df.coeff.shape[0],21
	V_computed = np.zeros((L_max,d_max))
	for d in range(0,d_max):
    	V_computed[:,d] = np.dot( df.coeff,d**np.arange(0,L_max,dtype=np.double) ) 
    diff = np.array((df.V_exact_int-V_computed),dtype=np.int)
    assert diff.sum()==0
"""
