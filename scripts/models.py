import torch
import torch_geometric
import transformers

import quantizer

class BaseCVAE(torch.nn.Module):

    def __init__(self, embedding, embedding_hidden_size, encoder_layer_sizes, latent_size, num_latent_embeddings, beta, decoder_layer_sizes, conditioner_layer_sizes, target_seq_length):
        super().__init__()

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        context_embedding_size = conditioner_layer_sizes[-1]

        self.latent_size = latent_size
        self.encoder = Encoder(embedding, embedding_hidden_size, encoder_layer_sizes, latent_size, context_embedding_size)
        self.decoder = Decoder(decoder_layer_sizes, latent_size, context_embedding_size, target_seq_length)
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = quantizer.VectorQuantizer(num_latent_embeddings, latent_size, beta)

    def forward(self, conditioner_context, target_input_ids=None, target_input_attention_mask=None, target_output_ids=None, latent=None, generate=False, **generate_kwargs):

        if generate:
            return self._generate(latent, conditioner_context, **generate_kwargs)

        condition = self.conditioner(conditioner_context)
        encoding = self.encoder(target_input_ids, target_input_attention_mask, condition)

        embedding_loss, latent, perplexity, _, _ = self.vector_quantization(encoding)

        logits, recon_loss = self.decoder(latent, condition, target_ids=target_output_ids)

        return logits, recon_loss, embedding_loss, latent, perplexity

    def reparameterize(self, mu, log_var):

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std

    def generate(self, latent, conditioner_context, **generate_kwargs):
        return self(conditioner_context, latent=latent, generate=True, **generate_kwargs)

    def _generate(self, latent, conditioner_context, **generate_kwargs):

        condition = self.conditioner(conditioner_context)
        outputs = self.decoder.generate(latent, condition, **generate_kwargs)

        return outputs

def get_embedding():
    config = transformers.DistilBertConfig.from_pretrained("distilbert-base-uncased")
    embedding_hidden_size = config.hidden_size
    embedding = transformers.DistilBertModel.from_pretrained("distilbert-base-uncased", config=config)

    # freeze weights
    for param in embedding.parameters():
        param.requires_grad = False

    return embedding, embedding_hidden_size

class CVAE(BaseCVAE):
    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes, conditioner_layer_sizes, target_seq_length):

        embedding, embedding_hidden_size = get_embedding()

        super().__init__(embedding, embedding_hidden_size, encoder_layer_sizes, latent_size, decoder_layer_sizes, conditioner_layer_sizes, target_seq_length)

        self.conditioner = Conditioner(embedding, embedding_hidden_size, conditioner_layer_sizes)

class GCVAE(BaseCVAE):
    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes, conditioner_layer_sizes, target_seq_length):
        
        embedding, embedding_hidden_size = get_embedding()
        
        super().__init__(embedding, embedding_hidden_size, encoder_layer_sizes, latent_size, decoder_layer_sizes, conditioner_layer_sizes, target_seq_length)

        self.conditioner = GraphConditioner(embedding, embedding_hidden_size, conditioner_layer_sizes)


class GraphConditioner(torch.nn.Module):

    def __init__(self, embedding, embedding_hidden_size, graph_layer_sizes):
        super().__init__()

        self.embedding = embedding

        graph_layers = []

        for i, (in_size, out_size) in enumerate(zip([embedding_hidden_size] + graph_layer_sizes[:-1], graph_layer_sizes)):
            graph_layers.append((torch_geometric.nn.GATConv(in_size, out_size), 'x, edge_index -> x'))
            graph_layers.append((torch.nn.ReLU(inplace=True)))
            # TODO add batch norm here ?

        self.graph_sequential_model = torch_geometric.nn.Sequential('x, edge_index', graph_layers)

        gate_nn = torch.nn.Linear(graph_layer_sizes[-1], 1)
        self.pooling_layer = torch_geometric.nn.GlobalAttention(gate_nn)

    def forward(self, input):
        output = self.embedding(input_ids=input.input_ids, attention_mask=input.attention_mask)
        # get average of sequence
        x = torch.mean(output.last_hidden_state, 1)
        x = self.graph_sequential_model(x, input.edge_index)
        return self.pooling_layer(x, batch=input.input_ids_batch, size=input.num_graphs)

class Conditioner(torch.nn.Module):

    def __init__(self, embedding, embedding_hidden_size, layer_sizes):
        super().__init__()
        
        self.embedding = embedding

        self.layers = torch.nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip([embedding_hidden_size] + layer_sizes[:-1], layer_sizes)):
            self.layers.add_module(name=f"L{i}", module=torch.nn.Linear(in_size, out_size))
            self.layers.add_module(name=f"A{i}", module=torch.nn.ReLU())
            self.layers.add_module(name=f"D{i}", module=torch.nn.Dropout())

    def forward(self, input):
        input_ids, attention_mask = input
        output = self.embedding(input_ids=input_ids, attention_mask=attention_mask)
        x = torch.mean(output.last_hidden_state, 1)
        return self.layers(x)


class Encoder(torch.nn.Module):

    def __init__(self, embedding, embedding_hidden_size, layer_sizes, latent_size, context_embedding_size):

        super().__init__()

        self.embedding = embedding

        self.MLP = torch.nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip([embedding_hidden_size + context_embedding_size] + layer_sizes, layer_sizes + [latent_size])):
            self.MLP.add_module(name=f"L{i}", module=torch.nn.Linear(in_size, out_size))
            self.MLP.add_module(name=f"A{i}", module=torch.nn.ReLU())
            self.MLP.add_module(name=f"D{i}", module=torch.nn.Dropout())

    def forward(self, input_ids, attention_mask, condition):
        output = self.embedding(input_ids=input_ids, attention_mask=attention_mask)
        # get verage of final hidden states
        x = torch.mean(output.last_hidden_state, 1)

        x = torch.cat((x, condition), dim=-1)

        x = self.MLP(x)

        return x


class Decoder(torch.nn.Module):

    def __init__(self, layer_sizes, latent_size, context_embedding_size, target_seq_length):
        super().__init__()

        self.target_seq_length = target_seq_length

        config = transformers.T5Config.from_pretrained("google/t5-efficient-tiny")
        lm_hidden_size = config.d_model

        self.MLP = torch.nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip([latent_size + context_embedding_size] + layer_sizes, layer_sizes + [lm_hidden_size])):
            self.MLP.add_module(name=f"L{i}", module=torch.nn.Linear(in_size, out_size))
            self.MLP.add_module(name=f"A{i}", module=torch.nn.ReLU())
            self.MLP.add_module(name=f"D{i}", module=torch.nn.Dropout())

        self.lm = transformers.T5ForConditionalGeneration.from_pretrained("google/t5-efficient-tiny")
        
        # later weights
        for param in self.lm.lm_head.parameters():
            param.requires_grad = False

        for param in self.lm.decoder.block[2:].parameters():
            param.requires_grad = False

    def forward(self, latent, condition, target_ids=None, generate=False, **generate_kwargs):
        z = torch.cat((latent, condition), dim=-1)

        x = self.MLP(z)
        # TODO create other options?
        # investigate unpooling layers
        x = x.unsqueeze(1).expand(-1, self.target_seq_length, -1)

        if generate:
            encoder_outputs = transformers.modeling_outputs.BaseModelOutput(last_hidden_state=x)
            return self.lm.generate(encoder_outputs=encoder_outputs, **generate_kwargs)
        else:
            output = self.lm(encoder_outputs=(x,), labels=target_ids)
            return output.logits, output.loss

    def generate(self, latent, condition, **generate_kwargs):
        return self(latent, condition, generate=True, **generate_kwargs)