import torch
import torch_geometric
import transformers


class BaseCVAE(torch.nn.Module):

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes, conditioner_layer_sizes):
        super().__init__()

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size
        self.encoder = Encoder(encoder_layer_sizes, latent_size)
        self.decoder = Decoder(decoder_layer_sizes, latent_size)

    def forward(self, y, z=None, x=None):

        if z and not x:
            return self.inference(z, y)

        y = self.conditioner(y)
        means, log_var = self.encoder(x, y)
        z = self.reparameterize(means, log_var)
        logits, loss = self.decoder(z, y)

        return logits, loss, means, log_var, z

    def reparameterize(self, mu, log_var):

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std

    def inference(self, z, y):

        y = self.conditioner(y)
        logits, loss = self.decoder(z, y)

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

        self.graph_layers = torch_geometric.nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip([roberta_hidden_size] + graph_layer_sizes[:-1], graph_layer_sizes)):
            self.graph_layers.add_module(name=f"L{i}", module=torch_geometric.nn.GATConv(in_size, out_size))
            self.graph_layers.add_module(name=f"A{i}", module=torch.nn.ReLU())

        gate_nn = torch.nn.Linear(graph_layer_sizes[-1], 1)
        self.pooling_layer = torch_geometric.nn.GlobalAttention(gate_nn)

    def forward(self, input):
        x = self.embedding(input.x)
        x = self.graph_layers(x, input.edge_index)
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

    def forward(self, x):
        x = self.embedding(x)
        return self.layers(x)


class Encoder(torch.nn.Module):

    def __init__(self, layer_sizes, latent_size, context_embedding_size, num_labels):

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

    def forward(self, x, c):
        x = self.embedding(x)

        x = torch.cat((x, c), dim=-1)

        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Decoder(torch.nn.Module):

    def __init__(self, layer_sizes, latent_size, context_embedding_size, num_labels):
        super().__init__()

        config = transformers.GPT2Config.from_pretrained("gpt2")
        text_gen_hidden_size = config.n_embd

        self.MLP = torch.nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip([latent_size + context_embedding_size] + layer_sizes, layer_sizes + [text_gen_hidden_size])):
            self.MLP.add_module(name=f"L{i}", module=torch.nn.Linear(in_size, out_size))
            self.MLP.add_module(name=f"A{i}", module=torch.nn.ReLU())
            self.MLP.add_module(name=f"D{i}", module=torch.nn.Dropout())

        self.text_gen = transformers.GPT2LMHeadModel.from_pretrained("gpt2", config=config)

    def forward(self, z, c):
        z = torch.cat((z, c), dim=-1)

        x = self.MLP(z)

        output = self.text_gen(encoder_hidden_states=x)

        return output.logits, output.loss