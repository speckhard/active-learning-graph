# This script is supposed to work.
# There is no catch beyond code quality.
# With default args (32, 32) the inference should take ~1min on CPU.
# If this script does not work, get back to Entalpic.
# You should delete these comments in your refactor.
import argparse

import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import DimeNetPlusPlus
from tqdm import tqdm

from entalpic_al import HOME, TARGET

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for the inference loop"
    )
    parser.add_argument(
        "--num_batches", type=int, default=32, help="Number batches to test on"
    )
    args = parser.parse_args()
    print("Using batch size", args.batch_size)
    dataset = QM9(HOME)
    # DimeNet uses the atomization energy for targets U0, U, H, and G, i.e.:
    # 7 -> 12, 8 -> 13, 9 -> 14, 10 -> 15
    idx = torch.tensor([0, 1, 2, 3, 4, 5, 6, 12, 13, 14, 15, 11])
    dataset.data.y = dataset.data.y[:, idx]
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model, datasets = DimeNetPlusPlus.from_qm9_pretrained(HOME, dataset, TARGET)
    _, _, test_dataset = datasets
    model = model.to(device)
    loader = DataLoader(test_dataset, batch_size=args.batch_size)
    mae_pretrained = []
    for d, data in enumerate(tqdm(loader, total=args.num_batches)):
        data = data.to(device)
        with torch.no_grad():
            pred = model(data.z, data.pos, data.batch)
        mae = (pred.view(-1) - data.y[:, TARGET]).abs()
        mae_pretrained.append(mae)
        if d == args.num_batches - 1:
            break
    mae_pretrained = torch.cat(mae_pretrained, dim=0)
    mae_pretrained = (mae_pretrained).mean().item()
    model.reset_parameters()
    mae_random = []
    for d, data in enumerate(tqdm(loader, total=args.num_batches)):
        data = data.to(device)
        with torch.no_grad():
            pred = model(data.z, data.pos, data.batch)
        mae = (pred.view(-1) - data.y[:, TARGET]).abs()
        mae_random.append(mae)
        if d == args.num_batches - 1:
            break
    mae_random = torch.cat(mae_random, dim=0)
    mae_random = (mae_random).mean().item()
    print(f"Pretrained MAE : {mae_pretrained:.4f} eV")
    print("Random init MAE: {mae_random} eV".format(mae_random=mae_random))
