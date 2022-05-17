import numpy as np
import pytest

from dadapy.id_discrete import IdDiscrete


def test_distances():
    """Test the discrete id estimator with canonical distances storing"""

    N = 500
    box = 20
    d = 2
    rng = np.random.default_rng(12345)

    X = rng.integers(0, box, size=(N, d))

    I3D = IdDiscrete(X, maxk=X.shape[0])
    I3D.compute_distances(metric="manhattan", period=box, condensed=False)

    I3D.compute_id_binomial_k(k=25, shell=False, ratio=0.5)
    assert I3D.intrinsic_dim == pytest.approx(2.047335150414252)

    I3D.compute_id_binomial_k(k=4, shell=True, ratio=0.5)
    assert I3D.intrinsic_dim == pytest.approx(2.014968714680048)

    I3D.compute_id_binomial_lk(
        lk=4, ln=2, method="bayes", plot=False, subset=np.arange(100)
    )
    assert I3D.intrinsic_dim == pytest.approx(1.9951144462611552)

    pv = I3D.model_validation_full(cdf=False)
    assert pv > 0.005


def test_distances_condensed():
    """Test the discrete id estimator with cumulative distances storing"""

    N = 500
    box = 20
    d = 2
    rng = np.random.default_rng(12345)

    X = rng.integers(0, box, size=(N, d))

    I3D = IdDiscrete(X, maxk=X.shape[0])
    I3D.compute_distances(metric="manhattan", period=box, condensed=True, d_max=d * box)

    I3D.compute_id_binomial_k(k=25, shell=False, ratio=0.5)
    assert I3D.intrinsic_dim == pytest.approx(2.047335150414252)

    I3D.compute_id_binomial_k(k=4, shell=True, ratio=0.5)
    assert I3D.intrinsic_dim == pytest.approx(2.014968714680048)

    I3D.compute_id_binomial_lk(lk=4, ln=2, method="mle")
    assert I3D.intrinsic_dim == pytest.approx(2.012434143811029)
