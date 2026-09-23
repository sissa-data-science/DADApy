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

"""
The *metric_comparisons* module contains the *MetricComparisons* class.

``MetricComparisons`` is preserved as a backward-compatible facade that combines
``InformationImbalance`` and ``NeighborhoodOverlap``. New code should prefer the
two focused classes directly.
"""

from dadapy.information_imbalance import InformationImbalance
from dadapy.neighborhood_overlap import NeighborhoodOverlap


class MetricComparisons(InformationImbalance, NeighborhoodOverlap):
    """Backward-compatible class combining InformationImbalance and NeighborhoodOverlap.

    Methods for comparing metric spaces are implemented on the two parent classes;
    this subclass exists so that existing code using ``MetricComparisons`` keeps
    working unchanged.
    """
