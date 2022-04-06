import collections
import copy
import os

import networkx as nx
import nltk
import numpy as np
import pandas as pd
import torch
import torch_geometric
import transformers
import tqdm

NUM_SENTENCES = 3
MAX_TOKENS = 300
MAX_TOP_PREDECESSORS = 3
NUM_GENERATIONS = 3

ContextTargetData = collections.namedtuple('ContextTargetData', ['target', 'context'])

def get_top_n_predecessors(graph, node, n):
    predecessors = graph.predecessors(node)
    predec_weights = []
    for predecessor in predecessors:
        predec_weights.append((predecessor, graph[predecessor][node]['entities']))

    sorted_predecessors = [predec[0] for predec in sorted(predec_weights, key=lambda x: x[1], reverse=True)]

    num_predecessors = graph.in_degree(node)
    return sorted_predecessors[:min(n, num_predecessors)]

def get_n_gen_ancestors(graph, node, num_gens, num_predecessors):
    if num_gens == 0:
        return set([node])

    ancestors = set()
    predecessors = get_top_n_predecessors(graph, node, num_predecessors)
    for predecessor in predecessors:
        sub_ancestors = get_n_gen_ancestors(graph, predecessor, num_gens - 1, num_predecessors)
        ancestors.add(predecessor)
        ancestors.update(sub_ancestors)

    return ancestors

class NewsDataset(torch_geometric.data.InMemoryDataset):
    def __init__(self, root, dataset_name='main', graph_context=True, transform=None, pre_transform=None, pre_filter=None):
        self.dataset_name = dataset_name
        self.graph_context = graph_context
        self.num_data_points = 0

        super().__init__(root, transform, pre_transform, pre_filter)

        data_save_path = os.path.join(self.processed_dir, 'data.pt')
        graph_data_save_path = os.path.join(self.processed_dir, 'graph_data.pt')

        self.data_list = torch.load(data_save_path)
        self.graph_data = torch.load(graph_data_save_path)

    def len(self):
        return len(self.data_list)

    def get(self, idx: int):

        if self.graph_context:
            graph_data, graph_slices = self.graph_data
            
            if not hasattr(self, '_graph_data_list') or self._graph_data_list is None:
                self._graph_data_list = self.len() * [None]
            elif self._graph_data_list[idx] is not None:
                data = copy.copy(self.data_list[idx])
                data['conditioner_graph'] = copy.copy(self._graph_data_list[idx])
                return data

            graph_context = torch_geometric.data.separate.separate(
                cls=graph_data.__class__,
                batch=graph_data,
                idx=idx,
                slice_dict=graph_slices,
                decrement=False,
            )

            self._graph_data_list[idx] = copy.copy(graph_context)

            data = copy.copy(self.data_list[idx])
            data['conditioner_graph'] = copy.copy(self._graph_data_list[idx])
            return data
        else:
            return copy.copy(self.data_list[idx])

    @property
    def raw_file_names(self):
        return [f"{self.dataset_name}_entity_dag_{data_type}.csv" for data_type in ['nodes', 'edges']]

    @property
    def processed_file_names(self):
        return ['data.pt', 'graph_data.pt']

    def process(self):

        # load graph from file
        nodes_df = pd.read_csv(os.path.join(self.raw_dir, f"{self.dataset_name}_entity_dag_nodes.csv"))
        edges_df = pd.read_csv(os.path.join(self.raw_dir, f"{self.dataset_name}_entity_dag_edges.csv"))

        # drop nan rows
        nodes_df = nodes_df.dropna()

        graph = nx.DiGraph()

        roberta_tokenizer = transformers.RobertaTokenizerFast.from_pretrained("roberta-base")
        gpt2_tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
        # set pad_token_id to unk_token_id -> be careful here as unk_token_id == eos_token_id == bos_token_id
        gpt2_tokenizer.pad_token = gpt2_tokenizer.unk_token

        node_mapping = {}
        # add nodes from dataframe
        print('Loading nodes into graph')
        for idx, node_row in tqdm.tqdm(nodes_df.iterrows(), total=len(nodes_df)):
            node_mapping[node_row['id']] = idx

            first_sentences = ' '.join(nltk.tokenize.sent_tokenize(node_row['text'])[:NUM_SENTENCES - 1])
            node_text = node_row['title'] + '. ' + first_sentences
            roberta_tokens = roberta_tokenizer(node_text, padding='max_length', truncation=True, max_length=MAX_TOKENS)
            gpt2_tokens = gpt2_tokenizer(node_text, padding='max_length', truncation=True, max_length=MAX_TOKENS)

            graph.add_node(idx, 
                           roberta_input_ids=roberta_tokens.input_ids,
                           roberta_attention_mask=roberta_tokens.attention_mask,
                           gpt2_input_ids=gpt2_tokens.input_ids,
                           gpt2_attention_mask=gpt2_tokens.attention_mask)

        # add edges from dataframe
        print('Loading edges into graph')
        for idx, edge_row in tqdm.tqdm(edges_df.iterrows(), total=len(edges_df)):
            if edge_row['old_id'] not in node_mapping or edge_row['new_id'] not in node_mapping:
                continue

            old_id = node_mapping[edge_row['old_id']]
            new_id = node_mapping[edge_row['new_id']]

            if graph.has_edge(old_id, new_id):
                graph[old_id][new_id]['entities'] += 1
            else:
                graph.add_edge(old_id, new_id, entities=1)

        data_list = []
        graph_data_list = []

        # process network into torch compat shape
        print('Create context/target pairs from graph')
        for node, tokens in tqdm.tqdm(graph.nodes(data=True), total=graph.number_of_nodes()):
            if graph.in_degree(node) == 0:
                continue

            data = {}

            data['target_input_ids'] = torch.tensor(tokens['roberta_input_ids'], dtype=torch.long)
            data['target_input_attention_mask'] = torch.tensor(tokens['roberta_attention_mask'], dtype=torch.long)

            data['target_output_ids'] = torch.tensor(tokens['gpt2_input_ids'], dtype=torch.long)
            data['target_output_attention_mask'] = torch.tensor(tokens['gpt2_attention_mask'], dtype=torch.long)

            context_node = get_top_n_predecessors(graph, node, 1)[0]
            data['decoder_input_ids'] = torch.tensor(graph.nodes[context_node]['gpt2_input_ids'])
            data['decoder_attention_mask'] = torch.tensor(graph.nodes[context_node]['gpt2_attention_mask'])

            if self.graph_context:
                ancestors = get_n_gen_ancestors(graph, node, NUM_GENERATIONS, MAX_TOP_PREDECESSORS)
                graph_context = nx.induced_subgraph(graph, ancestors).copy()

                num_nodes = len(graph_context)
                node_token_ids = np.zeros((num_nodes, MAX_TOKENS))
                node_attention_mask = np.zeros((num_nodes, MAX_TOKENS))

                node_map = {}
                for node_idx, (context_node, context_tokens) in enumerate(graph_context.nodes(data=True)):
                    node_token_ids[node_idx, :] = context_tokens['roberta_input_ids']
                    node_attention_mask[node_idx, :] = context_tokens['roberta_attention_mask']

                    node_map[context_node] = node_idx

                edges = []
                for (node_1, node_2) in graph_context.edges():
                    edges.append([node_map[node_1], node_map[node_2]])
                
                graph_context = torch_geometric.data.Data(
                    input_ids = torch.tensor(node_token_ids, dtype=torch.long),
                    attention_mask = torch.tensor(node_attention_mask, dtype=torch.long),
                    edge_index = torch.tensor(edges, dtype=torch.long).T
                )

                graph_data_list.append(graph_context)

            else:
                data['conditioner_input_ids'] = torch.tensor(graph[context_node]['roberta_input_ids'])
                data['conditioner_attention_mask'] = torch.tensor(graph[context_node]['roberta_attention_mask'])

            data_list.append(data)


        data_save_path = os.path.join(self.processed_dir, 'data.pt')
        graph_data_save_path = os.path.join(self.processed_dir, 'graph_data.pt')

        torch.save(data_list, data_save_path)
        if self.graph_context:
            data, slices = self.collate(graph_data_list)
            torch.save((data, slices), graph_data_save_path)


def load_and_preprocess_dataset(model, dataset_name, batch_size):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', dataset_name)
    graph_context = model == 'gcvae'

    dataset = NewsDataset(path, dataset_name, graph_context=graph_context)

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    train_dataloader = torch_geometric.loader.DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = torch_geometric.loader.DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = torch_geometric.loader.DataLoader(test_dataset, batch_size=batch_size)

    return train_dataloader, val_dataloader, test_dataloader