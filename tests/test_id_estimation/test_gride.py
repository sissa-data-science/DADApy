import os

import numpy as np
import pytest

from dadapy import DensityEstimation

def test_compute_id_gride():
    """Test that the id estimations with gride work correctly"""
    filename = os.path.join(os.path.split(__file__)[0], "../2gaussians_in_2d.npy")

    np.random.seed(0)

    X = np.load(filename)

    de = DensityEstimation(coordinates=X)

    ### testing gride scaling
    #TODO: @diegodoimo there seem to be a bug here
    # and you are the best person to fix it :D
    ids, ids_err, rs = de.return_id_scaling_gride()

    # expected_ids = None
    # expected_ids_err = None
    # expected_rs = None
    #assert ids == pytest.approx(expected_ids, abs=0.01)
    #assert ids_err == pytest.approx(expected_ids_err, abs=0.01)
    #assert rs == pytest.approx(expected_rs, abs=0.01)
