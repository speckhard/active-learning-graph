"""Test that we can load data from qm9 into fully connected graphs."""

import os
import sys

# Append path so that we can access packages that are above this directory.
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import data_eng.qm9_to_fc_graphs as qm9_to_graphs


def test_fc_output_shape():
    """Test that graphs are fully connected."""
    train_data, _, _ = qm9_to_graphs.load_fc_graphs_from_qm9()

    num_nodes_in_graph = train_data[0].x.shape[0]
    # Check that the shapes are what we expect.
    assert (train_data[0].edge_index.shape[0] == 2,
            'Edge indices are given in COO format (sender/receiver style).')
    assert (
        train_data[0].edge_index.shape[1] == num_nodes_in_graph*(
            num_nodes_in_graph-1),
        'Fully connected graphs should have # atoms * # atoms -1,'
        'since its a permutation. # Atoms P 2, e.g. for 5 atoms, 5!/(5-2)!')

