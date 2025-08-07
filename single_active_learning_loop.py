"""Active learning loop """


import random
import os
import sys

import numpy as np
import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import DimeNetPlusPlus

from botorch import acquisition, sampling
from botorch.acquisition.objective import ScalarizedPosteriorTransform, ConstrainedMCObjective
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.box_decompositions import FastNondominatedPartitioning
from gpytorch.mlls import ExactMarginalLogLikelihood

from entalpic_al import HOME, TARGET

# Append path so that we can access packages that are above this directory.
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import models.gcn_model as gcn_model
import data_eng.qm9_to_fc_graphs as qm9_to_graphs


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # Not needed for CPU.
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dimenet_prediction(model, datapoint, device='cpu'):
    """Get a prediction of a dimenet model given a single datapoint."""
    model = model.to(device)  # Can remove these since running on CPU.
    datapoint = datapoint.to(device)
    with torch.no_grad():
        pred = model(datapoint.z, datapoint.pos, datapoint.batch)
    return pred


def get_dimenet_and_gcn_data(num_graphs: int = 2000):
    """Get data for the dimenet and the GCN model.
    
    The GCN model expects one hot encoded models and an edge index (COO format).
    We also fully connect the graphs in our setup.

    The DimeNet model expects atomic numbers and coordinates and performs
    it's own cutoff function to get the graph's connectivity. Code snippet
    below.
    ```
            edge_index = radius_graph(pos, r=self.cutoff, batch=batch,
                                  max_num_neighbors=self.max_num_neighbors)
    ```
    """

    # Note this should be the same source as the GCN model except the GCN model
    # runs transforms on the data.
    qm9_raw_dataset = QM9(HOME)

    data_ids = torch.randint(
        low=0, high=len(qm9_raw_dataset)-1, size=(num_graphs,)).numpy()

    # Training_data
    gcn_data = qm9_to_graphs.load_fc_graphs_from_qm9(
        num_graphs=num_graphs, data_ids = data_ids)

    dimenet_data = qm9_raw_dataset[data_ids]
    # DimeNet uses the atomization energy for targets U0, U, H, and G, i.e.:
    # 7 -> 12, 8 -> 13, 9 -> 14, 10 -> 15
    idx = torch.tensor([0, 1, 2, 3, 4, 5, 6, 12, 13, 14, 15, 11])
    dimenet_data.data.y = dimenet_data.data.y[:, idx]

    return gcn_data, dimenet_data


# def get_most_uncertain_id(model_list, remaining_id_list, data):
#     """Get the most uncertain id from a list of models."""
#     # We need to calculate the standard deviation of the model.
#     data = data[remaining_id_list]
#     # Get predictions on the entire dataset.
#     predictions_list = []
#     for model in model_list:
#         predictions = model(data)
#         predictions_list.append(predictions)
#     # Now stack the predictions.
#     stacked_predictions = torch.cat(predictions_list, axis=0)
#     print(stacked_predictions)
#     mean_prediction = 

@torch.no_grad()
def get_deep_ensemble_mean_and_stdev(model_list, data):
    """Get the mean and stdev predictions."""
    # We need to calculate the standard deviation of the model.
    # Get predictions on the entire dataset.
    predictions_list = []
    for model in model_list:
        predictions = model(data)
        predictions_list.append(predictions)
    # Now stack the predictions.
    stacked_predictions = torch.cat(predictions_list, axis=0)
    print(stacked_predictions)
    mean_prediction = torch.mean(stacked_predictions, dim=0)
    std_prediction = torch.std(stacked_predictions, dim=0)
    return mean_prediction, std_prediction


def train_from_scratch(
        list_of_seeds,
        input_dim,
        embedding_dim,
        output_dim,
        data,
        batch_size=32,
        num_epochs=1,
        property=14):
    """Retrain the model from scratch.
    Args:
        model_list: list of models to retrain.
        data: Data on which to retrain.
        TODO(dts): add the rest.
    """
    model_list = []
    for seed in list_of_seeds:
        set_seed(seed)
        
        model = gcn_model.GCN(input_dim=input_dim,
                        embedding_dim=embedding_dim,
                        output_dim=output_dim)
        model_list.append(model.fit_model(
            train_data=data, val_data=None,
            batch_size=batch_size, num_epochs=num_epochs,
            property=property))
    return model_list

def retrain_single_datapoint(
        model_list,
        data,
        property=14):
    """Train from a pretrained state with a single datapoint.
    Args:
        model_list: list of models to retrain.
        datapoint: Data on which to retrain.
    """
    for model in model_list:
        # Should retrain the models in place (list of pointers to models.)
        # TODO(dts): need to check this.
        model.fit_model(
            train_data=data, val_data=None,
            batch_size=1, num_epochs=1,
            property=property)
    return model_list

def run_deep_ensemble_active_learning_loop(
        num_active_loops: int = 10,
        num_learners_in_ensemble: int = 10,
        input_dim = 11,
        output_dim = 1,
        embedding_dim = 512,
        batch_size = 32,
        num_epochs = 10,
        from_scratch: bool = False):
    """Run the deep ensemble active learning loop on a GCN model.

    TODO(dts): make it model agnostic, so we can run on other models (e.g.
        GATv2, MPNN, PaiNN, MACE).
    
    Here a vanilla approach using the deep ensemble of the model is used.
    
    # We are told to use the DimeNet++ model as a labeler. I'm assuming
    # that this is not used for the initial 1000 training datapoints.
    # Might be wrong.

    Args:
        num_active_loops: Number of times to run the active learning loops.
        from_scratch: whether to re-train the model from scratch or train
            with an additional datapoint (online-learning).
    """
    # TODO(dts): change this to random ints when done testing.  
    set_seed(0)
    gcn_data, dimenet_data = get_dimenet_and_gcn_data(num_grapsh=3000)
    # Use the final 1000 for test.
    gcn_test_data = gcn_data[2000:]
    gcn_data = gcn_data[:2000]
    # TODO(dts): Some major assumptions that I don't have time to explore.
    # Is dimenet's data normalized like I am normalizing my data? If not,
    # normalize. Is the TARGET in the entalpic_al library the same target?
    print(f'Entalpic target is: {TARGET}')
    dimenet_model, _ = DimeNetPlusPlus.from_qm9_pretrained(
        HOME, dimenet_data, TARGET)

    # We reserve true data to 1k to train our GCN model. After that we
    # actively learn on our labeller model, the DimeNet++ model. 
    gcn_train_data = gcn_data[0:1000]
    # Alt. option to train on DimeNet directly.
    # gcn_train_data.y = dimenet_model(
    #     dimenet_data[0:1000].z, dimenet_data[0:1000].pos)
    list_of_seeds = np.arange(num_learners_in_ensemble)  # Make seeds random.
    # Get initial models using true data training data. See assumption
    # in docstring, this might be wrong.
    model_list = train_from_scratch(
        list_of_seeds=list_of_seeds,
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        output_dim=output_dim,
        train_data=gcn_train_data,
        val_data=None,
        batch_size=batch_size,
        num_epochs=num_epochs,
        property=14)

    id_list = np.arange(1000).tolist()
    remaining_id_list = np.arange(1000, 2000).tolist()

    for iteration in range(num_active_loops):
        # For every iteration find the most uncertain point.
        _, stdev = get_deep_ensemble_mean_and_stdev(
            model_list, gcn_data[remaining_id_list])
        id = torch.argmax(stdev)
        id_list.append(id)  # Add the datapoint to our training data.
        # Remove the datapoint from unexplored points.
        remaining_id_list.remove(id)
        if not remaining_id_list:
            # This list is empty and we cannot run anymore AL.
            break

        # Now make the DimeNet++ prediction on this datapoint
        datapoint = dimenet_data[id]
        dimenet_prediction = get_dimenet_prediction(list_of_seeds, datapoint)
        # Update the label with the dimenet prediction.
        gcn_data[id].y = dimenet_prediction
        # Now train the models either from scratch or not.

        if from_scratch:
            model_list = train_from_scratch(model_list, gcn_data[id_list])
        else:
            model_list = retrain_single_datapoint(model_list, gcn_data[id])
        
        if iteration % 10:
            # Get test results
            get_deep_ensemble_mean_and_stdev(model_list, gcn_test_data)
            print(f'iteration: {iteration}, ')


def run_bo_active_learning_loop(num_active_loops: int = 10,
        input_dim = 11,
        output_dim = 1,
        embedding_dim = 512,
        batch_size = 32,
        num_epochs = 10,
        from_scratch: bool = False):
    """Train a surrogate model to predict the uncertainty.
    
    We train a GP model to predict the difference between the GCN model
    and the DimeNet++ model.

    Inspired by tutorial here:
    https://www.physicsx.ai/newsroom/bayesian-optimization-and-active-learning-cookbook

    TODO(dts): Should make a parameter for batch optimization or sequential.
    """
    # Boiler plate code to get data and models. Should refactor since
    # we're copying it.
    set_seed(0)
    gcn_data, dimenet_data = get_dimenet_and_gcn_data(num_grapsh=3000)
    # Use the final 1000 for test.
    gcn_test_data = gcn_data[2000:]
    gcn_data = gcn_data[:2000]
    print(f'Entalpic target is: {TARGET}')
    dimenet_model, _ = DimeNetPlusPlus.from_qm9_pretrained(
        HOME, dimenet_data, TARGET)
    # We reserve true data to 1k to train our GCN model. After that we
    # actively learn on our labeller model, the DimeNet++ model. 
    gcn_train_data = gcn_data[0:1000]
    gcn_train_data.y = dimenet_model(
        dimenet_data[0:1000].z, dimenet_data[0:1000].pos)
    # Re-use this method from the deep ensemble method. Should be changed as
    # well since it's clunky here since we only hae one model.
    gcn_model = train_from_scratch(
        list_of_seeds=[0],
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        output_dim=output_dim,
        train_data=gcn_train_data,
        val_data=None,
        batch_size=batch_size,
        num_epochs=num_epochs,
        property=14)[0]
    
    gcn_train_predictions = gcn_model(gcn_train_data)

    # Note, gcn_train_data.y is already the dimenet predictions.
    gp = SingleTaskGP(gcn_train_predictions, gcn_train_data.y) 
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    # UCB = mean(x) + sqrt(beta)*var(x)
    # beta controls exploration vs exploitation.
    # In this case we want our model to get better so let's set beta
    # to a high number.
    # TODO(dts): try the probability of improvement here as a
    # different acquisition function.
    ucb = acquisition.UpperConfidenceBound(gp, beta=10)
    for iteration in num_active_loops:
        # TODO(dts):
        # What I would do if I had more time.
        # evaluate the UCB on all the datapoints left out.
        # Then label the one with the largest acquisition function
        # using dimenet. Get the prediction from the GCN model.
        # Then retrain the GP from scratch or train a new single datapoint.

        # With even more time I would create a multi-fidelity loop.
        # Assign a cost to the different surrogates/methods to label data
        # (e.g. DimeNet++ and another model e.g. Orb). Train on all levels
        # of fidelity (true data, surrogates). Acquisition function
        # decides what level of fidelity to use next based on associated costs.
        # Example, adapting the UCB acquisition function to a surrogate and true
        # values:
        # lambda(true)*mean(true) + sqrt(beta_1)var(true) + \
        #   lambda(surrogate)*mean(surrogate) + sqrt(beta_2)var(surrogate)
