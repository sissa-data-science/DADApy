import os

import numpy as np
import pytest

from dadapy import MetricComparisons

filename = os.path.join(os.path.split(__file__)[0], "../3d_gauss_small_z_var.npy")


def test_return_label_overlap_selected_coords():
    """Test that the label overlap works correctly"""

    X = np.load(filename)
    labels = np.ones(X.shape[0])
    labels[np.where(X[:, 0] < 0)] = 0
    coord_list = [[0, 1, 2], [0], [1], [2]]
    expected_overlaps = [0.7783333333333333, 0.927, 0.494, 0.48533333333333334]

    mc = MetricComparisons(coordinates=X, maxk=X.shape[0] - 1)

    mc.compute_distances()

    overlaps = mc.return_label_overlap_selected_coords(
        labels=labels, coord_list=coord_list
    )

    assert overlaps == pytest.approx(expected_overlaps)
