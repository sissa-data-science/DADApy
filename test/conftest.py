import random

import numpy as np

import pytest


@pytest.fixture(autouse=True)
def fixed_seed():
    """This fixture will make sure that the random seed is the same for all tests"""
    random.seed(0)
    np.random.seed(0)
