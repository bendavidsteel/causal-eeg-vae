import torch
import torch_geometric
import transformers

import gpt2

class BaseCVAE(torch.nn.Module):

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes, conditioner_layer_sizes):
        super().__init__()

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        context_embedding_size = conditioner_layer_sizes[-1]

        self.latent_size = latent_size
        self.encoder = Encoder(encoder_layer_sizes, latent_size, context_embedding_size)
        self.decoder = Decoder(decoder_layer_sizes, latent_size, context_embedding_size)

    def forward(self, conditioner_context, decoder_input_ids, decoder_attention_mask, target_input_ids=None, target_attention_mask=None, latent=None):

        if latent and not target_input_ids:
            self.inference(latent, conditioner_context, decoder_input_ids, decoder_attention_mask)

        condition = self.conditioner(conditioner_context)
        means, log_var = self.encoder(target_input_ids, target_attention_mask, condition)
        latent = self.reparameterize(means, log_var)
        logits = self.decoder(latent, decoder_input_ids, decoder_attention_mask, condition)

        return logits, means, log_var, latent

    def reparameterize(self, mu, log_var):

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std

    def inference(self, latent, conditioner_context, decoder_input_ids, decoder_attention_mask):

        condition = self.conditioner(conditioner_context)
        logits = self.decoder(latent, decoder_input_ids, decoder_attention_mask, condition)

        return logits

class CVAE(BaseCVAE):
    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes, conditioner_layer_sizes):
        super().__init__(encoder_layer_sizes, latent_size, decoder_layer_sizes, conditioner_layer_sizes)

        self.conditioner = Conditioner(conditioner_layer_sizes)

class GCVAE(BaseCVAE):
    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes, conditioner_layer_sizes):
        super().__init__(encoder_layer_sizes, latent_size, decoder_layer_sizes, conditioner_layer_sizes)

        self.conditioner = GraphConditioner(conditioner_layer_sizes)


class GraphConditioner(torch.nn.Module):

    def __init__(self, graph_layer_sizes):
        super().__init__()
        config = transformers.RobertaConfig.from_pretrained("roberta-base")
        roberta_hidden_size = config.hidden_size
        self.embedding = transformers.RobertaModel.from_pretrained("roberta-base", config=config)

        graph_layers = []

        for i, (in_size, out_size) in enumerate(zip([roberta_hidden_size] + graph_layer_sizes[:-1], graph_layer_sizes)):
            graph_layers.append((torch_geometric.nn.GATConv(in_size, out_size), 'x, edge_index -> x'))
            graph_layers.append((torch.nn.ReLU(inplace=True)))

        self.graph_sequential_model = torch_geometric.nn.Sequential('x, edge_index', graph_layers)

        gate_nn = torch.nn.Linear(graph_layer_sizes[-1], 1)
        self.pooling_layer = torch_geometric.nn.GlobalAttention(gate_nn)

    def forward(self, input):
        output = self.embedding(input_ids=input.input_ids, attention_mask=input.attention_mask)
        # get last hidden state at end of sequence
        x = output.last_hidden_state[:,-1,:]
        x = self.graph_sequential_model(x, input.edge_index)
        return self.pooling_layer(x)

class Conditioner(torch.nn.Module):

    def __init__(self, layer_sizes):
        super().__init__()
        config = transformers.RobertaConfig.from_pretrained("roberta-base")
        roberta_hidden_size = config.hidden_size
        self.embedding = transformers.RobertaModel.from_pretrained("roberta-base", config=config)

        self.layers = torch.nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip([roberta_hidden_size] + layer_sizes[:-1], layer_sizes)):
            self.layers.add_module(name=f"L{i}", module=torch.nn.Linear(in_size, out_size))
            self.layers.add_module(name=f"A{i}", module=torch.nn.ReLU())

    def forward(self, input):
        input_ids, attention_mask = input
        x = self.embedding(input_ids=input_ids, attention_mask=attention_mask)
        return self.layers(x)


class Encoder(torch.nn.Module):

    def __init__(self, layer_sizes, latent_size, context_embedding_size):

        super().__init__()

        config = transformers.RobertaConfig.from_pretrained("roberta-base")
        input_embedding_size = config.hidden_size
        self.embedding = transformers.RobertaModel.from_pretrained("roberta-base", config=config)

        self.MLP = torch.nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip([input_embedding_size + context_embedding_size] + layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(name=f"L{i}", module=torch.nn.Linear(in_size, out_size))
            self.MLP.add_module(name=f"A{i}", module=torch.nn.ReLU())
            self.MLP.add_module(name=f"D{i}", module=torch.nn.Dropout())

        self.linear_means = torch.nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = torch.nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, input_ids, attention_mask, condition):
        output = self.embedding(input_ids=input_ids, attention_mask=attention_mask)
        # get last hidden state at end of sequence
        x = output.last_hidden_state[:,-1,:]

        x = torch.cat((x, condition), dim=-1)

        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Decoder(torch.nn.Module):

    def __init__(self, layer_sizes, latent_size, context_embedding_size):
        super().__init__()

        config = transformers.GPT2Config.from_pretrained("gpt2")
        text_gen_hidden_size = config.n_embd

        self.MLP = torch.nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip([latent_size + context_embedding_size] + layer_sizes, layer_sizes + [text_gen_hidden_size])):
            self.MLP.add_module(name=f"L{i}", module=torch.nn.Linear(in_size, out_size))
            self.MLP.add_module(name=f"A{i}", module=torch.nn.ReLU())
            self.MLP.add_module(name=f"D{i}", module=torch.nn.Dropout())

        self.gpt2 = gpt2.GPT2LMHeadModel.from_pretrained("gpt2", config=config)

    def forward(self, latent, input_ids, attention_mask, condition):
        z = torch.cat((latent, condition), dim=-1)

        x = self.MLP(z)

        output = self.gpt2(input_ids=input_ids, attention_mask=attention_mask, latent_variable=x)

        return output.logits