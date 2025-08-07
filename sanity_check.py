"""Refactor the sanity check code.

This code runs a pretrained DimeNet++ on QM9 data.

Open TODO(dts):
1. Clean up the code in the model evaluate loop (we shouldn't infer one datapoint at a time)
2. Clean up the code in the get model and datasets so that 

"""

import logging

from absl import flags
from absl import app

import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import DimeNetPlusPlus
from tqdm import tqdm

from entalpic_al import HOME, TARGET


FLAGS = flags.FLAGS
flags.DEFINE_integer(
    'batch_size',
    '32',  # Default value if not specified.
    'What batch size to use')
flags.DEFINE_integer(
    'num_batches',
    '2',  # Default value if not specified.
    'Number of batches to evaluate on.')
flags.DEFINE_string(
    'device',
    'cpu',  # Having issues running on the GPU on my personal laptop.
    'What device to run inference on.')


class ModelEvaluator():
    def __init__(self, model, datasets, device):
        """Initialize model evaluation with a model and datset.
        
        Args:
            model: PyTorch model ready for evaluation.
            datasets: Tuple of training/validation/test datasets.
            device: Which device to run the inference on.
        """
        self.device = device
        self.model = model.to(self.device)
        _, _, self.test_dataset = datasets  # We only need the test data.
    
    @torch.no_grad()
    def evaluate_model(self, batch_size, total_num_batches):
        """Evaluate the model on the test data.
        
        Note, I thought it might make more sense to simply evaluate on the
        entire dataset. E.g.:

        ```
            test_predictions = self.model(
            self.test_dataset.z, self.test_dataset.pos, None)
            abs_error = torch.abs(
                test_predictions.view(-1) - self.test_dataset.y[:, TARGET])
        ```
        This is however very slow on CPU machine and now I realize why the
        total number of batches is given.
        """

        loader = DataLoader(self.test_dataset, batch_size=batch_size)
        abs_error_list = []

        for batch_number, data in enumerate(tqdm(loader, total=total_num_batches)):
            data = data.to(self.device)
            pred = self.model(data.z, data.pos, data.batch)
            abs_error = (pred.view(-1) - data.y[:, TARGET]).abs()
            abs_error_list.append(abs_error)
            if batch_number == total_num_batches - 1:
                break
        abs_error = torch.cat(abs_error_list, dim=0)
        mae_metric = (abs_error).mean().item()
        return mae_metric


def get_qm9_data_and_model(pretrained: bool = True):
    dataset = QM9(HOME)
    # DimeNet uses the atomization energy for targets U0, U, H, and G, i.e.:
    # 7 -> 12, 8 -> 13, 9 -> 14, 10 -> 15
    idx = torch.tensor([0, 1, 2, 3, 4, 5, 6, 12, 13, 14, 15, 11])
    dataset.data.y = dataset.data.y[:, idx]

    model, datasets = DimeNetPlusPlus.from_qm9_pretrained(HOME, dataset, TARGET)

    if pretrained == False:  # Then randomize the weights of the model.
        model.reset_parameters()
    return model, datasets


def main(argv):
    batch_size = FLAGS.batch_size
    num_batches = FLAGS.num_batches
    device = FLAGS.device
    logger = logging.getLogger(__name__)
    logger.info(f'Using batch size: {batch_size}')
    logger.info(f'Using {num_batches} batches to evaluate.')
    logger.info(f'Running on {device}.')
    
    for pretrained in [True, False]:
        model, datasets = get_qm9_data_and_model(pretrained)
        model_evaluator = ModelEvaluator(model, datasets, device)
        mae_metric = model_evaluator.evaluate_model(batch_size, num_batches)
        logger.info(f'DimeNet++, Pretrained: {pretrained}, MAE: {mae_metric:.4f}')


if __name__ == '__main__':
    app.run(main)
