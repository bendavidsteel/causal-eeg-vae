import argparse
import os.path as osp

import torch
import torchtext
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

from scripts import cvae, models

MAX_LOGSTD = 10
MAX_EPOCHS = 1000

def kl_loss(mu, logstd):
    r"""Computes the KL loss, either for the passed arguments :obj:`mu`
    and :obj:`logstd`, or based on latent variables from last encoding.

    Args:
        mu (Tensor): The latent space for :math:`\mu`. If set to
            :obj:`None`, uses the last computation of :math:`mu`.
            (default: :obj:`None`)
        logstd (Tensor): The latent space for
            :math:`\log\sigma`.  If set to :obj:`None`, uses the last
            computation of :math:`\log\sigma^2`.(default: :obj:`None`)
    """
    logstd = logstd.clamp(max=MAX_LOGSTD)
    return -0.5 * torch.mean(torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))

def train(model, train_data, val_data):

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()

    min_val_loss = None

    for epoch in range(MAX_EPOCHS):

        # train on training set
        train_loss = 0
        for batch in train_data:
            optimizer.zero_grad()

            logits, recon_loss, means, log_var, z = model(train_data.context, x=train_data.targets)
            loss = recon_loss + kl_loss(means, log_var)
            loss.backward()
            optimizer.step()
            train_loss += float(loss)

        # trail on validation set
        val_loss = 0
        for batch_idx, batch in enumerate(val_data):
            with torch.no_grad():
                logits, recon_loss, means, log_var, z = model(train_data.context, x=train_data.targets)
            loss = recon_loss + kl_loss(means, log_var)

            val_loss += float(loss)

        if not min_val_loss:
            min_val_loss = val_loss

        # save best model so far
        if val_loss < min_val_loss:
            # checkpoint model

            epochs_since_best = 0
        else:
            epochs_since_best += 1

        # early stopping
        if epochs_since_best > 100:
            break

        print(f'Epoch: {epoch:03d}, AUC: {auc:.4f}, AP: {ap:.4f}')


def test(model, data):
    model.eval()
    with torch.no_grad():
        logits, recon_loss, means, log_var, z = model.encode(data.context)

    

    torchtext.data.metrics.bleu_score(, max_n=2)

    return model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)


def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data, val_data, test_data = load_and_preprocess_dataset(device, train=True)

    if args.model == 'gcvae':
        model = models.GCVAE()
    elif args.model == 'cvae':
        model = models.CVAE()

    model = model.to(device)

    train(model, train_data, val_data)
    test(model, test_data)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gcvae', choices=['gcvae', 'cvae'])
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    args = parser.parse_args()

    main(args)