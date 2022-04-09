import os

import numpy as np
import pytest

from dadapy import MetricComparisons

filename = os.path.join(os.path.split(__file__)[0], "../3d_gauss_small_z_var.npy")


def test_return_label_overlap():
    """Test that the label overlap works correctly"""

    X1 = np.load(filename)
    X2 = X1 + 1.0  # shifted gaussian

    X = np.vstack((X1, X2))  # datasets with two Gaussians

    # the labels distinguish the two Gaussians
    labels = np.ones(X.shape[0])
    labels[: X1.shape[0]] = 0

    mc = MetricComparisons(coordinates=X)
    mc.compute_distances()
    overlap = mc.return_label_overlap(labels=labels)

    assert overlap == 0.8676666666666668


def test_return_label_overlap_selected_coords():
    """Test that the label overlap works correctly"""

    X = np.load(filename)

    # labels simply distinguish left from right
    labels = np.ones(X.shape[0])
    labels[np.where(X[:, 0] < 0)] = 0

    coord_list = [[0, 1, 2], [0], [1], [2]]
    # coordinate 0 is expected to better distinguish left and right
    expected_overlaps = [0.7783333333333333, 0.927, 0.494, 0.48533333333333334]

    mc = MetricComparisons(coordinates=X, maxk=X.shape[0] - 1)

    mc.compute_distances()

    overlaps = mc.return_label_overlap_selected_coords(
        labels=labels, coord_list=coord_list
    )

    assert overlaps == pytest.approx(expected_overlaps)
