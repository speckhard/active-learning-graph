"""Let's create a GCN model with pytorch layers.

Let's create a very simple graph convoluational model that follows
Kipf and Welling's 2016 papr. We choose this model since it does
not have many parameters which is a requirement (<1 million)
and will therefore train quickly allowing us to validate our
pipeline.
"""

import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import torch.nn.functional as f
from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool import global_mean_pool
import torch_geometric.transforms as T
from torch_geometric.utils import remove_self_loops, to_dense_adj, dense_to_sparse

class GCN(torch.nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int, output_dim: int):
        super().__init__()
        self.embedding_layer = torch.nn.Linear(input_dim, embedding_dim)
        self.gcn_layer_one = GCNConv(embedding_dim, embedding_dim)
        self.gcn_layer_two = GCNConv(embedding_dim, embedding_dim)
        self.pool = global_mean_pool
        self.lin_pred = torch.nn.Linear(embedding_dim, output_dim)
    
    def forward(self, x, edge_index, batch=None):
        """Forward pass of the model."""
        # TODO(dts): if there is more time, make the # of layers a
        # configurable parameters and we can run a for loop over the number
        # of layers. Right now the model architecture is rigid.
        embedded_features = self.embedding_layer(x)
        hidden_one = self.gcn_layer_one(embedded_features, edge_index)
        hidden_one = torch.relu(hidden_one)
        hidden_two = self.gcn_layer_two(hidden_one, edge_index)
        hidden_two = torch.relu(hidden_two)
        pooled_result = self.pool(hidden_two, batch)
        output = self.lin_pred(pooled_result)
        return output
    
    def fit_model(
            self, train_data, val_data = None,
            batch_size: int = 32, num_epochs: int = 1,
            property: int = 14):
        """Fitting the model to the training data."""
        # Use the MSE since it is smooth compared to the MAE.
        criterion = torch.nn.MSELoss()
        # Note the weights given here are arbitrary and need to be learned.
        optimizer = torch.optim.Adam(
            self.parameters(), lr=0.01, weight_decay=5e-4)
        self.train()

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

        for epoch in range(num_epochs):
            for batched_data in train_loader:
                # Zero the gradients from the last step to avoid accumulation.
                optimizer.zero_grad()

                train_prediction = self(batched_data.x, batched_data.edge_index)
                train_mse_loss = criterion(train_prediction, train_data.y[:, property])
                train_mse_loss.backward()
                optimizer.step()

            if (epoch % 10 == 0):
                if val_data is not None:
                    val_prediction = self(
                        val_data.x, val_data.edge_index)
                    val_mse_loss = criterion(
                        val_prediction, val_data.y[:, property])
                else:
                    val_mse_loss = 'N/A'
                # Get train stats on all of the training data.
                full_train_prediction = self(
                    train_data.x, train_data.edge_index)
                full_train_mse_loss = criterion(
                    full_train_prediction, train_data.y[:, property])
                print(f'Epoch: {epoch}, training MSE: {train_mse_loss:4f}')
                print(f'Epoch: {epoch}, validation MSE: {val_mse_loss:4f}')

            # TODO(dts): this is where we would implement early stopping.

        return full_train_mse_loss, val_mse_loss
    
    @torch.no_grad()
    def test_model(self, test_data, property: int = 14):
        self.eval()
        test_predictions = self(test_data.x, test_data.edge_index)
        test_mse = torch.nn.MSELoss()(test_predictions, test_data.y[:, :, property])
        return test_mse

