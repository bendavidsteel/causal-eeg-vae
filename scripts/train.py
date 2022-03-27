

import argparse
import itertools
import os.path as osp

import torch

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GAE, VGAE, GCNConv


class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

class GCNConditioner(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


def train(model, train_data, optimizer):
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)
    loss = model.recon_loss(z, train_data.pos_edge_label_index)
    if args.variational:
        loss = loss + (1 / train_data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(model, data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    return model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)


def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                        split_labels=True, add_negative_train_samples=False),
    ])
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
    dataset = Planetoid(path, args.dataset, transform=transform)
    train_data, val_data, test_data = dataset[0]

    in_channels, out_channels = dataset.num_features, 16

    autoencoder = VGAE(VariationalGCNEncoder(in_channels, out_channels))
    conditioner = GCNConditioner(in_channels, out_channels)

    autoencoder = autoencoder.to(device)
    conditioner = conditioner.to(device)
    parameters = itertools.chain(autoencoder.parameters(), conditioner.parameters())
    optimizer = torch.optim.Adam(parameters, lr=0.01)

    for epoch in range(1, args.epochs + 1):
        loss = train(autoencoder, conditioner, train_data, optimizer)
        auc, ap = test(autoencoder, conditioner, test_data)
        print(f'Epoch: {epoch:03d}, AUC: {auc:.4f}, AP: {ap:.4f}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora',
                        choices=['Cora', 'CiteSeer', 'PubMed'])
    parser.add_argument('--epochs', type=int, default=400)
    args = parser.parse_args()

    main(args)