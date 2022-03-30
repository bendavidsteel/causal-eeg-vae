
import os

import torch
import torch_geometric

def load_and_preprocess_dataset(device):
    transform = torch_geometric.transforms.Compose([
        torch_geometric.transforms.ToDevice(device),
        torch_geometric.transforms.RandomNodeSplit(num_val=0.05, num_test=0.1),
    ])
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'Planetoid')
    dataset = Planetoid(path, args.dataset, transform=transform)

    return dataset[0]