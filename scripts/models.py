import torch
import torch_geometric
import transformers

import gpt2
import quantizer

class BaseCVAE(torch.nn.Module):

    def __init__(self, embedding, embedding_hidden_size, encoder_layer_sizes, latent_size, num_latent_embeddings, beta, decoder_layer_sizes, conditioner_layer_sizes, target_seq_length):
        super().__init__()

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        context_embedding_size = conditioner_layer_sizes[-1]

        self.latent_size = latent_size
        
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = quantizer.VectorQuantizer(num_latent_embeddings, latent_size, beta)

    def forward(self, conditioner_context, target_input_ids=None, target_input_attention_mask=None, target_output_ids=None, latent=None, generate=False, prompt=None, **generate_kwargs):

        if generate:
            return self._generate(latent, conditioner_context, prompt=prompt, **generate_kwargs)

        condition = self.conditioner(conditioner_context)
        latent = self.encoder(target_input_ids, target_input_attention_mask, condition)

        embedding_loss, latent_quantized, perplexity, _, _ = self.vector_quantization(latent)

        logits, recon_loss = self.decoder(latent_quantized, condition, target_ids=target_output_ids)

        return logits, recon_loss, embedding_loss, perplexity

    def reparameterize(self, mu, log_var):

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std

    def generate(self, latent, conditioner_context, prompt=None, **generate_kwargs):
        return self(conditioner_context, latent=latent, generate=True, prompt=prompt, **generate_kwargs)

    def _generate(self, latent, conditioner_context, prompt=None, **generate_kwargs):

        _, latent_quantized, _, _, _ = self.vector_quantization(latent)
        condition = self.conditioner(conditioner_context)
        outputs = self.decoder.generate(latent_quantized, condition, prompt=prompt, **generate_kwargs)

        return outputs

def get_embedding():
    config = transformers.DistilBertConfig.from_pretrained("distilbert-base-uncased")
    embedding_hidden_size = config.hidden_size
    embedding = transformers.DistilBertModel.from_pretrained("distilbert-base-uncased", config=config)

    # freeze weights
    for param in embedding.embeddings.parameters():
        param.requires_grad = False

    for param in embedding.transformer.layer[:4].parameters():
        param.requires_grad = False

    return embedding, embedding_hidden_size

class CVAE(BaseCVAE):
    def __init__(self, lm_name, encoder_layer_sizes, latent_size, num_latent_embeddings, beta, decoder_layer_sizes, conditioner_layer_sizes, target_seq_length):

        embedding, embedding_hidden_size = get_embedding()

        super().__init__(embedding, embedding_hidden_size, encoder_layer_sizes, latent_size, num_latent_embeddings, beta, decoder_layer_sizes, conditioner_layer_sizes, target_seq_length)

        self.conditioner = Conditioner(embedding, embedding_hidden_size, conditioner_layer_sizes)
        context_embedding_size = conditioner_layer_sizes[-1]
        self.encoder = SequenceEncoder(embedding, embedding_hidden_size, encoder_layer_sizes, latent_size, embedding_hidden_size)
        self.decoder = SequenceDecoder(decoder_layer_sizes, latent_size, embedding_hidden_size, target_seq_length, lm_name)

class GCVAE(BaseCVAE):
    def __init__(self, lm_name, encoder_layer_sizes, latent_size, num_latent_embeddings, beta, decoder_layer_sizes, conditioner_layer_sizes, target_seq_length):
        
        embedding, embedding_hidden_size = get_embedding()
        
        super().__init__(embedding, embedding_hidden_size, encoder_layer_sizes, latent_size, num_latent_embeddings, beta, decoder_layer_sizes, conditioner_layer_sizes, target_seq_length)

        self.conditioner = GraphConditioner(embedding, embedding_hidden_size, conditioner_layer_sizes)
        context_embedding_size = conditioner_layer_sizes[-1]
        self.encoder = Encoder(embedding, embedding_hidden_size, encoder_layer_sizes, latent_size, context_embedding_size)
        self.decoder = Decoder(decoder_layer_sizes, latent_size, context_embedding_size, target_seq_length, lm_name)


class GraphConditioner(torch.nn.Module):

    def __init__(self, embedding, embedding_hidden_size, graph_layer_sizes):
        super().__init__()

        self.embedding = embedding
        self.attn = torch.nn.Linear(embedding_hidden_size, 1)

        graph_layers = []

        for in_size, out_size in zip([embedding_hidden_size] + graph_layer_sizes[:-1], graph_layer_sizes):
            graph_layers.append((torch_geometric.nn.GATConv(in_size, out_size), 'x, edge_index -> x'))
            graph_layers.append((torch_geometric.nn.LayerNorm(out_size), 'x, batch -> x'))
            graph_layers.append((torch.nn.ReLU(inplace=True)))

        self.graph_sequential_model = torch_geometric.nn.Sequential('x, edge_index, batch', graph_layers)

        gate_nn = torch.nn.Linear(graph_layer_sizes[-1], 1)
        self.pooling_layer = torch_geometric.nn.GlobalAttention(gate_nn)

    def forward(self, input):
        output = self.embedding(input_ids=input.input_ids, attention_mask=input.attention_mask)
        # use attention to pool weights
        attn_weights = self.attn(output.last_hidden_state)
        x = torch.bmm(output.last_hidden_state.permute(0, 2, 1), attn_weights).squeeze(2)

        x = self.graph_sequential_model(x, input.edge_index, input.input_ids_batch)
        return self.pooling_layer(x, batch=input.input_ids_batch, size=input.num_graphs)

class Conditioner(torch.nn.Module):

    def __init__(self, embedding, embedding_hidden_size, layer_sizes):
        super().__init__()
        
        self.embedding = embedding

    def forward(self, input):
        input_ids, attention_mask = input
        output = self.embedding(input_ids=input_ids, attention_mask=attention_mask)

        return output.last_hidden_state


class Encoder(torch.nn.Module):

    def __init__(self, embedding, embedding_hidden_size, layer_sizes, latent_size, context_embedding_size):

        super().__init__()

        self.embedding = embedding

        self.MLP = torch.nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip([embedding_hidden_size + context_embedding_size] + layer_sizes, layer_sizes + [latent_size])):
            self.MLP.add_module(name=f"L{i}", module=torch.nn.Linear(in_size, out_size))
            self.MLP.add_module(name=f"N{i}", module=torch.nn.LayerNorm(out_size))
            self.MLP.add_module(name=f"A{i}", module=torch.nn.ReLU())
            self.MLP.add_module(name=f"D{i}", module=torch.nn.Dropout())

        self.attn = torch.nn.Linear(embedding_hidden_size, 1)

    def forward(self, input_ids, attention_mask, condition):
        output = self.embedding(input_ids=input_ids, attention_mask=attention_mask)
        # use attention to pool weights
        attn_weights = self.attn(output.last_hidden_state)
        x = torch.bmm(output.last_hidden_state.permute(0, 2, 1), attn_weights).squeeze(2)

        x = torch.cat((x, condition), dim=-1)

        x = self.MLP(x)

        return x


class SequenceEncoder(torch.nn.Module):

    def __init__(self, embedding, embedding_hidden_size, layer_sizes, latent_size, context_embedding_size):

        super().__init__()

        self.embedding = embedding

        self.MLP = torch.nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip([embedding_hidden_size + context_embedding_size] + layer_sizes, layer_sizes + [latent_size])):
            self.MLP.add_module(name=f"L{i}", module=torch.nn.Linear(in_size, out_size))
            self.MLP.add_module(name=f"N{i}", module=torch.nn.LayerNorm(out_size))
            self.MLP.add_module(name=f"A{i}", module=torch.nn.ReLU())
            self.MLP.add_module(name=f"D{i}", module=torch.nn.Dropout())

        self.attn = torch.nn.Linear(embedding_hidden_size, 1)
        self.condition_attn = torch.nn.Linear(context_embedding_size, 1)

    def forward(self, input_ids, attention_mask, condition):
        output = self.embedding(input_ids=input_ids, attention_mask=attention_mask)
        # use attention to pool weights
        attn_weights = self.attn(output.last_hidden_state)
        x = torch.bmm(output.last_hidden_state.permute(0, 2, 1), attn_weights).squeeze(2)

        cond_attn_weights = self.condition_attn(condition)
        cond = torch.bmm(condition.permute(0, 2, 1), cond_attn_weights).squeeze(2)

        x = torch.cat((x, cond), dim=-1)

        x = self.MLP(x)

        return x


class Decoder(torch.nn.Module):

    def __init__(self, layer_sizes, latent_size, context_embedding_size, target_seq_length, lm_name):
        super().__init__()

        self.target_seq_length = target_seq_length

        self.lm_name = lm_name
        if lm_name == 't5':
            config = transformers.T5Config.from_pretrained("google/t5-efficient-large")
            self.lm_hidden_size = config.d_model
            self.lm = transformers.T5ForConditionalGeneration.from_pretrained("google/t5-efficient-large")

            # later weights
            for param in self.lm.lm_head.parameters():
                param.requires_grad = False

            for param in self.lm.decoder.block[2:].parameters():
                param.requires_grad = False

        elif lm_name == 'gpt2':
            config = transformers.GPT2Config.from_pretrained("distilgpt2")
            self.lm_hidden_size = config.n_embd

            self.lm = gpt2.GPT2LMHeadModel.from_pretrained("distilgpt2", config=config)
            # freeze embedding weights
            for param in self.lm.transformer.wte.parameters():
                param.requires_grad = False

            # freeze position weights
            for param in self.lm.transformer.wpe.parameters():
                param.requires_grad = False

            # freeze later transformer blocks
            for param in self.lm.transformer.h[1:].parameters():
                param.requires_grad = False

            # freeze lm head
            for param in self.lm.lm_head.parameters():
                param.requires_grad = False

        self.MLP = torch.nn.Sequential()

        self.lstm_num_layers = 2

        for i, (in_size, out_size) in enumerate(zip([latent_size + context_embedding_size] + layer_sizes, layer_sizes + [self.lm_hidden_size])):
            self.MLP.add_module(name=f"L{i}", module=torch.nn.Linear(in_size, out_size))
            self.MLP.add_module(name=f"N{i}", module=torch.nn.LayerNorm(out_size))
            self.MLP.add_module(name=f"A{i}", module=torch.nn.ReLU())
            self.MLP.add_module(name=f"D{i}", module=torch.nn.Dropout())

        self.lstm = torch.nn.LSTM(self.lm_hidden_size, self.lm_hidden_size, self.lstm_num_layers, batch_first=True)
        

    def forward(self, latent, condition, target_ids=None, target_attention_mask=None, generate=False, prompt=None, **generate_kwargs):
        z = torch.cat((latent, condition), dim=-1)

        x = self.MLP(z)

        lstm_input_output = x.unsqueeze(1)

        # TODO implement multi head attention

        device = latent.device
        num_batch = latent.size(0)
        lstm_output = torch.zeros((num_batch, self.target_seq_length, self.lm_hidden_size)).to(device)
        lstm_hidden_state = torch.zeros((self.lstm_num_layers, num_batch, self.lm_hidden_size)).to(device)
        lstm_cell_state = torch.zeros((self.lstm_num_layers, num_batch, self.lm_hidden_size)).to(device)
        for seq_idx in range(self.target_seq_length):
            lstm_input_output, (lstm_hidden_state, lstm_cell_state) = self.lstm(lstm_input_output, (lstm_hidden_state, lstm_cell_state))
            lstm_output[:, seq_idx, :] = lstm_input_output.squeeze(1)

        if self.lm_name == 'gpt2':
            if generate:
                return self.lm.generate(input_ids=prompt, latent_variable=lstm_output, **generate_kwargs)
            else:
                output = self.lm(input_ids=target_ids, attention_mask=target_attention_mask, labels=target_ids, latent_variable=lstm_output)
                return output.logits, output.loss
        elif self.lm_name == 't5':
            if generate:
                encoder_outputs = transformers.modeling_outputs.BaseModelOutput(last_hidden_state=lstm_output)
                return self.lm.generate(encoder_outputs=encoder_outputs, **generate_kwargs)
            else:
                output = self.lm(encoder_outputs=(lstm_output,), labels=target_ids)
                return output.logits, output.loss

    def generate(self, latent, condition, prompt=None, **generate_kwargs):
        return self(latent, condition, generate=True, prompt=prompt, **generate_kwargs)



class SequenceDecoder(torch.nn.Module):

    def __init__(self, layer_sizes, latent_size, context_embedding_size, target_seq_length, lm_name):
        super().__init__()

        self.target_seq_length = target_seq_length

        self.lm_name = lm_name
        if lm_name == 't5':
            config = transformers.T5Config.from_pretrained("google/t5-efficient-large")
            self.lm_hidden_size = config.d_model
            self.lm = transformers.T5ForConditionalGeneration.from_pretrained("google/t5-efficient-large")

            # later weights
            for param in self.lm.lm_head.parameters():
                param.requires_grad = False

            for param in self.lm.decoder.block[2:].parameters():
                param.requires_grad = False

        elif lm_name == 'gpt2':
            config = transformers.GPT2Config.from_pretrained("distilgpt2")
            self.lm_hidden_size = config.n_embd

            self.lm = gpt2.GPT2LMHeadModel.from_pretrained("distilgpt2", config=config)
            # freeze embedding weights
            for param in self.lm.transformer.wte.parameters():
                param.requires_grad = False

            # freeze position weights
            for param in self.lm.transformer.wpe.parameters():
                param.requires_grad = False

            # freeze later transformer blocks
            for param in self.lm.transformer.h[1:].parameters():
                param.requires_grad = False

            # freeze lm head
            for param in self.lm.lm_head.parameters():
                param.requires_grad = False

        self.MLP = torch.nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip([latent_size] + layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(name=f"L{i}", module=torch.nn.Linear(in_size, out_size))
            self.MLP.add_module(name=f"N{i}", module=torch.nn.LayerNorm(out_size))
            self.MLP.add_module(name=f"A{i}", module=torch.nn.ReLU())
            self.MLP.add_module(name=f"D{i}", module=torch.nn.Dropout())

        self.final_linear = torch.nn.Linear(layer_sizes[-1] + context_embedding_size, self.lm_hidden_size)
        

    def forward(self, latent, condition, target_ids=None, target_attention_mask=None, generate=False, prompt=None, **generate_kwargs):
        
        latent = self.MLP(latent)
        
        z = torch.cat((latent.unsqueeze(1).expand(-1, condition.size(1), -1), condition), dim=-1)

        x = self.final_linear(z)
        x = torch.nn.functional.relu(x)

        if self.lm_name == 'gpt2':
            if generate:
                return self.lm.generate(input_ids=prompt, latent_variable=x, **generate_kwargs)
            else:
                output = self.lm(input_ids=target_ids, attention_mask=target_attention_mask, labels=target_ids, latent_variable=x)
                return output.logits, output.loss
        elif self.lm_name == 't5':
            if generate:
                encoder_outputs = transformers.modeling_outputs.BaseModelOutput(last_hidden_state=x)
                return self.lm.generate(encoder_outputs=encoder_outputs, **generate_kwargs)
            else:
                output = self.lm(encoder_outputs=(x,), labels=target_ids)
                return output.logits, output.loss

    def generate(self, latent, condition, prompt=None, **generate_kwargs):
        return self(latent, condition, generate=True, prompt=prompt, **generate_kwargs)