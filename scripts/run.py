import argparse
import os

import nltk
import torch
import transformers

import models, dataset

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

def train(model, train_data, val_data, device, checkpoint_path, resume):

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()

    min_val_loss = None

    if resume:
        model.load_state_dict(torch.load(os.path.join(checkpoint_path, 'best_model.pt')))

    for epoch in range(MAX_EPOCHS):

        # train on training set
        train_loss = 0
        for batch in train_data:
            context = batch[0].to(device)
            target = batch[1].to(device)
            inference_start = torch.tensor([], dtype='long').to(device)

            optimizer.zero_grad()

            logits, recon_loss, means, log_var, z = model(context, inference_start, target=target)
            loss = recon_loss + kl_loss(means, log_var)
            loss.backward()
            optimizer.step()
            train_loss += float(loss)

        # trail on validation set
        val_loss = 0
        for batch in val_data:
            context = batch[0].to(device)
            target = batch[1].to(device)
            inference_start = torch.tensor([], dtype='long').to(device)

            with torch.no_grad():
                logits, recon_loss, means, log_var, z = model(context, inference_start, target=target)
            loss = recon_loss + kl_loss(means, log_var)

            val_loss += float(loss)

        if not min_val_loss:
            min_val_loss = val_loss

        # save best model so far
        if val_loss < min_val_loss:
            # checkpoint model
            torch.save(model.state_dict(), os.path.join(checkpoint_path, 'best_model.pt'))

            epochs_since_best = 0
        else:
            epochs_since_best += 1

        # early stopping
        if epochs_since_best > 100:
            break

        print(f'Epoch: {epoch:03d}, AUC: {auc:.4f}, AP: {ap:.4f}')


def test(model, data, device):

    tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")

    bleu_scores = []
    model.eval()
    with torch.no_grad():
        for batch in data:
            context = batch[0].to(device)
            target = batch[1].to(device)
            inference_start = torch.tensor([], dtype='long').to(device)

            latent_size = model.latent_size
            z = torch.nn.randn([1, latent_size]).to(device)
            logits, recon_loss, means, log_var, z = model.encode(context, inference_start, z=z)

            inferences = []
            for logits_single in logits:
                inference = tokenizer.convert_ids_to_tokens(logits_single)
                inferences.append(inference)

            for target, inference in zip(target, inference):
                score = nltk.translate.bleu_score.sentence_bleu([target], inferences, weights=[0.5, 0.5])
                bleu_scores.append(score)

    avg_bleu_score = sum(bleu_scores) / len(bleu_scores)
    print(f"Average bleu score is {avg_bleu_score}")


def gen(model, data, device):
    tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")

    bleu_scores = []
    model.eval()
    with torch.no_grad():
        for batch in data:
            context = batch[0].to(device)
            target = batch[1].to(device)
            inference_start = torch.tensor([], dtype='long').to(device)

            latent_size = model.latent_size
            z = torch.nn.randn([1, latent_size]).to(device)
            logits, recon_loss, means, log_var, z = model.encode(context, inference_start, z=z, target=target)

            inferences = tokenizer.batch_decode(logits)

    print(inferences)


def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data, val_data, test_data = dataset.load_and_preprocess_dataset(args.model, args.dataset, args.batch_size)

    encoder_layer_sizes = [256, 256]
    latent_size = 16
    decoder_layer_sizes = [256, 256]
    conditioner_layer_sizes = [256, 256, 256]
    if args.model == 'gcvae':
        model = models.GCVAE(encoder_layer_sizes, latent_size, decoder_layer_sizes, conditioner_layer_sizes)
    elif args.model == 'cvae':
        model = models.CVAE(encoder_layer_sizes, latent_size, decoder_layer_sizes, conditioner_layer_sizes)

    model = model.to(device)

    if args.mode == 'train':
        train(model, train_data, val_data, device, args.checkpoint_path, args.resume)
    elif args.mode == 'test':
        test(model, test_data, device)
    elif args.mode == 'gen':
        gen(model, test_data, device)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gcvae', choices=['gcvae', 'cvae'])
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'gen'])
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--checkpoint-path', type=str)
    parser.add_argument('--resume', type=bool)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    main(args)