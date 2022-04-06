import argparse
import os

import nltk
import torch
import transformers
import tqdm

import models, dataset

MAX_LOGSTD = 10
MAX_EPOCHS = 1000

def get_kl_loss(mu, logstd):
    logstd = logstd.clamp(max=MAX_LOGSTD)
    return -0.5 * torch.mean(torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))


def train(model, train_data, val_data, device, checkpoint_path, resume, graph_context):

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fct = torch.nn.CrossEntropyLoss()
    model.train()

    min_val_loss = 999999
    epochs_since_best = 0

    if resume:
        model.load_state_dict(torch.load(os.path.join(checkpoint_path, 'best_model.pt')))

    for epoch in range(MAX_EPOCHS):

        # train on training set
        train_loss = 0
        progress_bar_data = tqdm.tqdm(enumerate(train_data), total=len(train_data))
        for batch_idx, batch in progress_bar_data:
            target_input_ids = batch['target_input_ids'].to(device)
            target_input_attention_mask = batch['target_input_attention_mask'].to(device)

            target_output_ids = batch['target_output_ids'].to(device)
            target_output_attention_mask = batch['target_output_attention_mask'].to(device)

            decoder_input_ids = batch['decoder_input_ids'].to(device)
            decoder_attention_mask = batch['decoder_attention_mask'].to(device)

            if graph_context:
                conditioner_context = batch['conditioner_graph'].to(device)
            else:
                conditioner_context = (batch['conditioner_input_ids'].to(device), batch['conditioner_attention_mask'].to(device))

            optimizer.zero_grad()

            logits, means, log_var, z = model(conditioner_context, decoder_input_ids, decoder_attention_mask, 
                                                          target_input_ids=target_input_ids, target_attention_mask=target_input_attention_mask)
            
            recon_loss = loss_fct(logits.view(-1, logits.size(-1)), target_output_ids.view(-1))
            kl_loss = get_kl_loss(means, log_var)
            loss = recon_loss + kl_loss
            loss.backward()
            optimizer.step()
            batch_loss = float(loss)
            progress_bar_data.set_description(f"Current Loss: {batch_loss:.4f}")
            train_loss += batch_loss
        
        train_loss /= batch_idx

        # trail on validation set
        val_loss = 0
        for batch_idx, batch in enumerate(val_data):
            target_input_ids = batch['target_input_ids'].to(device)
            target_attention_mask = batch['target_input_attention_mask'].to(device)
            target_output_ids = batch['target_output_ids'].to(device)
            decoder_input_ids = batch['decoder_input_ids'].to(device)
            decoder_attention_mask = batch['decoder_attention_mask'].to(device)

            if graph_context:
                conditioner_context = batch['conditioner_graph'].to(device)
            else:
                conditioner_context = (batch['conditioner_input_ids'].to(device), batch['conditioner_attention_mask'].to(device))

            with torch.no_grad():
                logits, means, log_var, z = model(conditioner_context, decoder_input_ids, decoder_attention_mask, 
                                                              target_input_ids=target_input_ids, target_attention_mask=target_attention_mask)
            
            recon_loss = loss_fct(logits.view(-1, logits.size(-1)), target_output_ids.view(-1))
            kl_loss = get_kl_loss(means, log_var)
            loss = recon_loss + kl_loss

            val_loss += float(loss)

        val_loss /= batch_idx

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

        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')


def test(model, data, device, graph_context):

    tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")

    bleu_scores = []
    model.eval()
    with torch.no_grad():
        for batch in data:
            target_input_ids = batch['target_input_ids'].to(device)
            target_attention_mask = batch['target_attention_mask'].to(device)
            decoder_input_ids = batch['decoder_input_ids'].to(device)
            decoder_attention_mask = batch['decoder_attention_mask'].to(device)

            if graph_context:
                conditioner_context = batch['conditioner_graph'].to(device)
            else:
                conditioner_context = (batch['conditioner_input_ids'].to(device), batch['conditioner_attention_mask'].to(device))

            latent_size = model.latent_size
            z = torch.nn.randn([1, latent_size]).to(device)
            logits, recon_loss, means, log_var, z = model(conditioner_context, decoder_context, z=z)

            inferences = []
            for logits_single in logits:
                inference = tokenizer.convert_ids_to_tokens(logits_single)
                inferences.append(inference)

            for target, inference in zip(target, inference):
                score = nltk.translate.bleu_score.sentence_bleu([target], inferences, weights=[0.5, 0.5])
                bleu_scores.append(score)

    avg_bleu_score = sum(bleu_scores) / len(bleu_scores)
    print(f"Average bleu score is {avg_bleu_score}")


def gen(model, data, device, graph_context):
    tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")

    bleu_scores = []
    model.eval()
    with torch.no_grad():
        for batch in data:
            target_input_ids = batch['target_input_ids'].to(device)
            target_attention_mask = batch['target_attention_mask'].to(device)
            decoder_input_ids = batch['decoder_input_ids'].to(device)
            decoder_attention_mask = batch['decoder_attention_mask'].to(device)

            if graph_context:
                conditioner_context = batch['conditioner_graph'].to(device)
            else:
                conditioner_context = (batch['conditioner_input_ids'].to(device), batch['conditioner_attention_mask'].to(device))

            latent_size = model.latent_size
            z = torch.nn.randn([1, latent_size]).to(device)
            logits, recon_loss, means, log_var, z = model(conditioner_context, decoder_context, z=z, target=target)

            inferences = tokenizer.batch_decode(logits)

    print(inferences)


def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data, val_data, test_data = dataset.load_and_preprocess_dataset(args.model, args.dataset, args.batch_size)

    encoder_layer_sizes = [128, 128]
    latent_size = 4
    decoder_layer_sizes = [128, 128]
    conditioner_layer_sizes = [128, 128]
    if args.model == 'gcvae':
        model = models.GCVAE(encoder_layer_sizes, latent_size, decoder_layer_sizes, conditioner_layer_sizes)
    elif args.model == 'cvae':
        model = models.CVAE(encoder_layer_sizes, latent_size, decoder_layer_sizes, conditioner_layer_sizes)

    model = model.to(device)

    if args.mode == 'train':
        train(model, train_data, val_data, device, args.checkpoint_path, args.resume, args.model == 'gcvae')
    elif args.mode == 'test':
        test(model, test_data, device, args.model == 'gcvae')
    elif args.mode == 'gen':
        gen(model, test_data, device, args.model == 'gcvae')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gcvae', choices=['gcvae', 'cvae'])
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'gen'])
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints')
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    main(args)