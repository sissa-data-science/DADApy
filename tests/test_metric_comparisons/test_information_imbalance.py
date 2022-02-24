import os

import numpy as np
import pytest

from dadapy import MetricComparisons

filename = os.path.join(os.path.split(__file__)[0], "../3d_gauss_small_z_var.npy")


def test_information_imbalance_basics():
    """Test the information imbalance operations work correctly"""

    X = np.load(filename)

    mc = MetricComparisons(coordinates=X)

    mc.compute_distances()

    coord_list = [
        [0,],
        [1,],
        [2,],
        [0, 1],
        [0, 2],
        [1, 2],
    ]

    imbalances = mc.return_inf_imb_full_selected_coords(coord_list)

    expected_imb = np.array(
        [
            [0.14400000000000002, 0.15, 0.968, 0.02, 0.1426, 0.1492],
            [0.5978, 0.5128, 0.9695999999999999, 0.02, 0.6208, 0.5434],
        ]
    )

    # Check we get the expected answer
    assert (imbalances == expected_imb).all()


def test_greedy_feature_selection_full():
    """Test thst the information imbalance greedy optimisation works correctly"""

    expeted_coords = np.array([1, 0])
    expected_imbalances = np.array([[0.14, 0.15, 0.97], [0.6, 0.51, 0.97]])

    X = np.load(filename)

    mc = MetricComparisons(coordinates=X, maxk=X.shape[0] - 1)

    selected_coords, all_imbalances = mc.greedy_feature_selection_full(n_coords=2)

    assert (selected_coords == expeted_coords).all()

    assert all_imbalances[0] == pytest.approx(expected_imbalances, abs=0.01)

    os.remove("all_imbalances.npy")
    os.remove("selected_coords.txt")


def test_return_inf_imb_matrix_of_coords():

    X = np.load(filename)[:, [0, 1]]

    expected_matrix = np.array([[0.0, 1.02], [0.99, 0.0]])

    mc = MetricComparisons(coordinates=X, maxk=X.shape[0] - 1)

    matrix = mc.return_inf_imb_matrix_of_coords()

    assert matrix == pytest.approx(expected_matrix, abs=0.01)
