"""Test the active learning loop."""

import os
import sys
import numpy as np

# Append path so that we can access packages that are above this directory.
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import single_active_learning_loop as sall

def test_get_dimenet_and_gcn_data():
    gcn_data, dimenet_data = sall.get_dimenet_and_gcn_data(num_graphs = 100)
    assert gcn_data[0].x.shape[1] == 11
    assert dimenet_data[0].x.shape[1] == 11
    # Ensure that we get the same data in row index one.
    np.testing.assert_array_equal(
        (gcn_data[0].x[0, :] - dimenet_data[0].x[0, :]).numpy(),
        np.zeros(11))



