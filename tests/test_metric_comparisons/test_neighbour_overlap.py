# Copyright 2021-2023 The DADApy Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Module for testing the neighbourhood overlap methods."""

import os

import numpy as np
import pytest

from dadapy import NeighborhoodOverlap

filename = os.path.join(os.path.split(__file__)[0], "../2gaussians_in_2d_overlap.npy")


def test_return_label_overlap():
    """Test that the label overlap works correctly."""
    X = np.load(filename)

    mc = NeighborhoodOverlap(coordinates=X)
    labels = np.ones(X.shape[0], dtype=int)
    # 20 points centerd in -1 are labeled = 0, 20 points centered in 1 are labeled 1
    labels[X[:, 0] < 0] = 0
    overlap = mc.return_label_overlap(labels=labels, k=5, avg=True)
    assert overlap == pytest.approx(1.0, 0.001)

    labels[30:] = 2
    ov_label = mc.return_label_overlap(
        labels=labels, class_fraction=0.25, avg=True, weighted=False
    )
    assert pytest.approx(0.675, 0.001) == ov_label

    ov_label = mc.return_label_overlap(
        labels=labels, class_fraction=0.25, avg=True, weighted=True
    )
    assert pytest.approx(0.5666, 0.001) == ov_label


def test_return_data_overlap():
    """Test that the label overlap works correctly."""
    X = np.load(filename)
    mc = NeighborhoodOverlap(coordinates=X)
    ov_data = mc.return_data_overlap(coordinates=X[:, 1:], k=30, avg=True)
    assert pytest.approx(0.78, 0.001) == ov_data


def test_constructor_other_data_overlap():
    """Two-dataset constructor: NeighborhoodOverlap(X1, X2).return_data_overlap()."""
    X = np.load(filename)
    mc = NeighborhoodOverlap(X, X[:, 1:])
    ov_data = mc.return_data_overlap(k=30, avg=True)
    assert pytest.approx(0.78, 0.001) == ov_data


def test_constructor_labels_label_overlap():
    """Constructor labels: NeighborhoodOverlap(X, labels=y).return_label_overlap()."""
    X = np.load(filename)
    labels = np.ones(X.shape[0], dtype=int)
    labels[X[:, 0] < 0] = 0
    mc = NeighborhoodOverlap(X, labels=labels)
    overlap = mc.return_label_overlap(k=5, avg=True)
    assert overlap == pytest.approx(1.0, 0.001)


def test_constructor_both_other_and_labels():
    """Constructor accepts both `other` and `labels` on the same instance."""
    X = np.load(filename)
    labels = np.ones(X.shape[0], dtype=int)
    labels[X[:, 0] < 0] = 0
    mc = NeighborhoodOverlap(X, other=X[:, 1:], labels=labels)
    assert pytest.approx(0.78, 0.001) == mc.return_data_overlap(k=30, avg=True)
    assert pytest.approx(1.0, 0.001) == mc.return_label_overlap(k=5, avg=True)


def test_constructor_missing_inputs_raise():
    """Calls without a second dataset / labels raise a clear error."""
    X = np.load(filename)
    mc = NeighborhoodOverlap(coordinates=X)
    with pytest.raises(AssertionError, match="no second dataset provided"):
        mc.return_data_overlap(k=30)
    with pytest.raises(AssertionError, match="no labels provided"):
        mc.return_label_overlap(k=5)
