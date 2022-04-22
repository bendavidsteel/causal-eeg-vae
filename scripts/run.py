import argparse
import os

import nltk
import torch
import torch_geometric
import transformers
import tqdm

import models, dataset

MAX_LOGSTD = 10
MAX_EPOCHS = 1000
EARLY_STOPPING_PATIENCE = 100


def train(model, train_data, val_data, device, checkpoint_path, resume, graph_context):

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
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

            if graph_context:
                conditioner_context = batch['conditioner_graph'].to(device)
            else:
                conditioner_context = (batch['conditioner_input_ids'].to(device), batch['conditioner_attention_mask'].to(device))

            optimizer.zero_grad()

            logits, recon_loss, embedding_loss, perplexity = model(conditioner_context, 
                                                                   target_input_ids=target_input_ids, 
                                                                   target_attention_mask=target_input_attention_mask,
                                                                   target_output_ids=target_output_ids)
            
            loss = recon_loss + embedding_loss
            loss.backward()
            optimizer.step()
            batch_loss = float(loss)
            progress_bar_data.set_description(f"Current Loss: {batch_loss:.4f}")
            train_loss += batch_loss
        
        train_loss /= batch_idx

        # trail on validation set
        val_loss = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_data):
                target_input_ids = batch['target_input_ids'].to(device)
                target_attention_mask = batch['target_input_attention_mask'].to(device)
                target_output_ids = batch['target_output_ids'].to(device)

                if graph_context:
                    conditioner_context = batch['conditioner_graph'].to(device)
                else:
                    conditioner_context = (batch['conditioner_input_ids'].to(device), batch['conditioner_attention_mask'].to(device))

                
                logits, recon_loss, embedding_loss, perplexity = model(conditioner_context, 
                                                                       target_input_ids=target_input_ids, 
                                                                       target_attention_mask=target_attention_mask,
                                                                       target_output_ids=target_output_ids)
                
                loss = recon_loss + embedding_loss

                val_loss += float(loss)

        val_loss /= batch_idx

        # potentially update learning rate
        scheduler.step(val_loss)

        # save best model so far
        if val_loss < min_val_loss:
            # checkpoint model
            torch.save(model.state_dict(), os.path.join(checkpoint_path, 'best_model.pt'))
            min_val_loss = val_loss
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        # early stopping
        if epochs_since_best > EARLY_STOPPING_PATIENCE:
            break

        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')


def test(model, data, device, graph_context, checkpoint_path, lm_name, gen_method):

    tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
    model.load_state_dict(torch.load(os.path.join(checkpoint_path, 'best_model.pt')))

    bleu_scores = []
    model.eval()
    with torch.no_grad():
        for batch in data:
            target_input_ids = batch['target_input_ids'].to(device)
            target_attention_mask = batch['target_attention_mask'].to(device)

            if graph_context:
                conditioner_context = batch['conditioner_graph'].to(device)
            else:
                conditioner_context = (batch['conditioner_input_ids'].to(device), batch['conditioner_attention_mask'].to(device))

            latent_size = model.latent_size
            latent = torch.nn.randn([1, latent_size]).to(device)

            if gen_method == 'topk':
                model_outputs = model.generate(
                    latent,
                    conditioner_context,
                    prompt=lm_tokenizer.bos_token if lm_name == 'gpt2' else None,
                    do_sample=True, 
                    max_length=101, 
                    top_k=50
                )
            elif gen_method == 'beam':
                model_outputs = model.generate(
                    latent,
                    conditioner_context,
                    prompt=lm_tokenizer.bos_token if lm_name == 'gpt2' else None,
                    max_length=101, 
                    num_beams=5, 
                    no_repeat_ngram_size=2, 
                    num_return_sequences=5, 
                    early_stopping=True
                )

            inferences = []
            for logits_single in logits:
                inference = tokenizer.convert_ids_to_tokens(logits_single)
                inferences.append(inference)

            for target, inference in zip(target, inference):
                score = nltk.translate.bleu_score.sentence_bleu([target], inferences, weights=[0.5, 0.5])
                bleu_scores.append(score)

    avg_bleu_score = sum(bleu_scores) / len(bleu_scores)
    print(f"Average bleu score is {avg_bleu_score}")


def gen(model, data, device, graph_context, checkpoint_path, lm_name, gen_method):
    if lm_name == 't5':
        lm_tokenizer = transformers.T5TokenizerFast.from_pretrained("google/t5-efficient-tiny")
    elif lm_name == 'gpt2':
        lm_tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
    
    bert_tokenizer = transformers.DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model.load_state_dict(torch.load(os.path.join(checkpoint_path, 'best_model.pt')))

    model.eval()
    with torch.no_grad():
        for batch in data:
            target_input_ids = batch['target_input_ids'].to(device)
            decoder_input_ids = batch['decoder_input_ids'].to(device)

            if graph_context:
                conditioner_context = batch['conditioner_graph'].to(device)
            else:
                conditioner_context = (batch['conditioner_input_ids'].to(device), batch['conditioner_attention_mask'].to(device))

            batch_size = decoder_input_ids.shape[0]
            latent_size = model.latent_size
            latent = torch.normal(0, 1, size=[batch_size, latent_size]).to(device)
            
            if gen_method == 'topk':
                model_outputs = model.generate(
                    latent,
                    conditioner_context,
                    prompt=lm_tokenizer.bos_token if lm_name == 'gpt2' else None,
                    do_sample=True, 
                    max_length=101, 
                    top_k=50
                )
            elif gen_method == 'beam':
                model_outputs = model.generate(
                    latent,
                    conditioner_context,
                    prompt=lm_tokenizer.bos_token if lm_name == 'gpt2' else None,
                    max_length=101, 
                    num_beams=5, 
                    no_repeat_ngram_size=2, 
                    num_return_sequences=5, 
                    early_stopping=True
                )

            print("Context:\n" + 100 * '-')
            print(lm_tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True))
            print("Target:\n" + 100 * '-')
            print(bert_tokenizer.decode(target_input_ids[0], skip_special_tokens=True))
            print("Output:\n" + 100 * '-')
            for i, beam_output in enumerate(model_outputs):
                print(f"{i}: {lm_tokenizer.decode(beam_output, skip_special_tokens=True)}")

def main(args):

    # ensure checkpoint dir exists
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset, val_dataset, test_dataset = dataset.load_and_preprocess_dataset(args.model, args.dataset)

    encoder_layer_sizes = [2048, 2048, 2048]
    latent_size = 8
    num_latent_embeddings = 512
    beta = 0.25
    decoder_layer_sizes = [2048, 2048, 2048]
    conditioner_layer_sizes = [2048, 2048, 2048]
    target_sequence_length = 100
    if args.model == 'gcvae':
        model = models.GCVAE(args.lm_name, encoder_layer_sizes, latent_size, num_latent_embeddings, beta, decoder_layer_sizes, conditioner_layer_sizes, target_sequence_length)
    elif args.model == 'cvae':
        model = models.CVAE(args.lm_name, encoder_layer_sizes, latent_size, num_latent_embeddings, beta, decoder_layer_sizes, conditioner_layer_sizes, target_sequence_length)

    model = model.to(device)

    if args.mode == 'train':
        train_dataloader = torch_geometric.loader.DataLoader(train_dataset, batch_size=args.batch_size, follow_batch=['input_ids', 'attention_mask'])
        val_dataloader = torch_geometric.loader.DataLoader(val_dataset, batch_size=args.batch_size, follow_batch=['input_ids', 'attention_mask'])
    
        train(model, train_dataloader, val_dataloader, device, args.checkpoint_path, args.resume, args.model == 'gcvae')
    elif args.mode == 'test':
        test_dataloader = torch_geometric.loader.DataLoader(test_dataset, batch_size=args.batch_size, follow_batch=['input_ids', 'attention_mask'])

        test(model, test_dataloader, device, args.model == 'gcvae', args.checkpoint_path, args.lm_name, args.gen_method)
    elif args.mode == 'gen':
        test_dataloader = torch_geometric.loader.DataLoader(test_dataset, batch_size=1, follow_batch=['input_ids', 'attention_mask'])

        gen(model, test_dataloader, device, args.model == 'gcvae', args.checkpoint_path, args.lm_name, args.gen_method)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gcvae', choices=['gcvae', 'cvae'])
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'gen'])
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints')
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lm-name', type=str, default='gpt2')
    parser.add_argument('--gen-method', type=str, default='topk')
    args = parser.parse_args()

    main(args)