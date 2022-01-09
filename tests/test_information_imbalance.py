import os

import numpy as np

from dadapy import MetricComparisons


def test_information_imbalance_basics():
    """Test the information imbalance operations work correctly"""
    filename = os.path.join(os.path.split(__file__)[0], "3d_gauss_small_z_var.npy")

    X = np.load(filename)

    mc = MetricComparisons(coordinates=X)

    mc.compute_distances()

    coord_list = [
        [
            0,
        ],
        [
            1,
        ],
        [
            2,
        ],
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
