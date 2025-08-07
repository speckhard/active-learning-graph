"""Test that the GCN model is working well.

// To run use:
python3 -m pytest test/test_gcn_model.py
"""

import os
import sys

import pytest

import torch
# Append path so that we can access packages that are above this directory.
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import models.gcn_model as gcn_model
import data_eng.qm9_to_fc_graphs as qm9_to_graphs


def test_inference_output_shape():
    """Test that for inference the output shape is as expected."""
    input_dim = 10
    output_dim = 2
    batch_size = 32
    num_nodes = 5
    model = gcn_model.GCN(input_dim=input_dim,
                      embedding_dim=100,
                      output_dim=output_dim)

    fake_data = torch.rand(batch_size, num_nodes, input_dim)  # Random matrix, batch size x input dim.
    fake_edge_index = torch.randint(low=0, high=num_nodes-1, size=(2, 10))
    output = model(fake_data, fake_edge_index)
    assert output.shape[0] == batch_size, 'First dim. of output should be batch size.'
    assert output.shape[1] == output_dim, 'Second dim. of output should be batch size.'


def test_num_parameters():
    """Test that the number of parameters of the GCN model is reasonable."""
    input_dim = 11
    output_dim = 1
    embedding_dim = 512
    expected_parameter_count = 531969
    model = gcn_model.GCN(input_dim=input_dim,
                      embedding_dim=embedding_dim,
                      output_dim=output_dim)
    # Get the number of parameters in the model.
    parameter_count = sum(p.numel() for p in model.parameters())
    
    assert parameter_count == expected_parameter_count

@pytest.mark.slow
def test_fit_model_on_qm9_data():
    """Test that for inference the output shape is as expected for qm9 data.
    
    Note this test is an integration test and runs several epochs of training
    data. Should be skipped when debugging with a --runslow option.
    """
    input_dim = 11
    output_dim = 1
    batch_size = 32
    num_batches = 12
    embedding_dim = 10
    num_epochs = 12
    model = gcn_model.GCN(input_dim=input_dim,
                      embedding_dim=embedding_dim,
                      output_dim=output_dim)

    train_data, val_data, _ = qm9_to_graphs.load_fc_graphs_from_qm9()

    train_mse, val_mse = model.fit_model(
        train_data, val_data, batch_size, num_epochs, property=14)
    assert train_mse > 0  # TODO(dts): Ideally add better integration tests.
    assert val_mse > 0

# TODO(dts): Add a test for the test_model method.
