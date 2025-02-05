# Copyright 2021-2025 The DADApy Authors. All Rights Reserved.
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

"""Module for testing methods of the CausalGraph class."""

import os
import sys

import numpy as np
import pytest
from jax import config

config.update("jax_platform_name", "cpu")
filename = os.path.join(os.path.split(__file__)[0], "../rosslers_xtoy_trajectory.npy")


@pytest.mark.skipif(sys.version_info < (3, 9), reason="Requires python>=3.9")
def test_CausalGraph_optimization():
    """Test CausalGraph optimization function."""
    from dadapy import CausalGraph  # noqa: E402

    # generate test data
    traj = np.load(filename)

    expected_weights = [0.06, 0.14, 0.13, 0.07, 0.13, 0.14]
    expected_imbs_final = [0.32, 0.33, 0.49, 0.27, 0.36, 0.46]

    # train the DII
    num_samples = 50
    num_epochs = 10
    batches_per_epoch = 1
    k = 5
    time_lags = [5] # time lags tested

    g = CausalGraph(time_series=traj, seed=0) # object of the class CausalGraph

    weights_final, imbs_training, _, _ = (
        g.optimize_present_to_future(
            num_samples=num_samples,
            time_lags=time_lags,
            target_variables="all",
            save_weights=False,
            embedding_dim_present=1,
            embedding_dim_future=1,
            embedding_time=1,
            num_epochs=num_epochs,
            batches_per_epoch=batches_per_epoch,
            l1_strength=0.,
            point_adapt_lambda=True,
            k_init=k,
            k_final=k,
            lambda_factor=0.1,
            params_init=None,
            optimizer_name="adam",
            learning_rate=1e-2,
            learning_rate_decay="cos",
            num_points_rows=None,
            compute_imb_final=False,
            compute_error=False,
            ratio_rows_columns=None,
            discard_close_ind=None
        )
    )

    assert weights_final[:,0,0] == pytest.approx(expected_weights, abs=0.01)
    assert imbs_training[:,0,-1] == pytest.approx(expected_imbs_final, abs=0.01)