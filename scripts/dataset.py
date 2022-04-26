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
MAX_TOKENS = 60
MAX_TOP_PREDECESSORS = 3
NUM_GENERATIONS = 3
MIN_COMMON_ENTITIES = 3

ContextTargetData = collections.namedtuple('ContextTargetData', ['target', 'context'])

def get_top_n_valid_predecessors(graph, node, n):
    predecessors = graph.predecessors(node)
    predec_weights = []
    for predecessor in predecessors:
        num_entities = graph[predecessor][node]['entities']
        if num_entities >= MIN_COMMON_ENTITIES:
            predec_weights.append((predecessor, num_entities))

    sorted_predecessors = [predec[0] for predec in sorted(predec_weights, key=lambda x: x[1], reverse=True)]

    num_predecessors = graph.in_degree(node)
    return sorted_predecessors[:min(n, num_predecessors)]

def get_n_gen_ancestors(graph, node, num_gens, num_predecessors):
    if num_gens == 0:
        return set([node])

    ancestors = set()
    predecessors = get_top_n_valid_predecessors(graph, node, num_predecessors)
    for predecessor in predecessors:
        sub_ancestors = get_n_gen_ancestors(graph, predecessor, num_gens - 1, num_predecessors)
        ancestors.add(predecessor)
        ancestors.update(sub_ancestors)

    return ancestors

class NewsDataset(torch_geometric.data.InMemoryDataset):
    def __init__(self, root, dataset_name='main', graph_context=True, lm_name='gpt2'):
        self.dataset_name = dataset_name
        self.graph_context = graph_context
        self.lm_name = lm_name
        self.num_data_points = 0

        super().__init__(root, None, None, None)

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
        return [file for file in os.listdir(self.raw_dir) if 'nodes' in file or 'edges' in file]

    @property
    def processed_file_names(self):
        return ['data.pt', 'graph_data.pt']

    def process(self):

        bert_tokenizer = transformers.DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

        if self.lm_name == 't5':
            lm_tokenizer = transformers.T5TokenizerFast.from_pretrained("google/t5-efficient-tiny")
        elif self.lm_name == 'gpt2':
            lm_tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
            # set pad_token_id to unk_token_id -> be careful here as unk_token_id == eos_token_id == bos_token_id
            lm_tokenizer.pad_token = lm_tokenizer.unk_token

        node_file_names = [file for file in os.listdir(self.raw_dir) if 'nodes' in file]

        data_list = []
        graph_data_list = []

        for node_file_name in node_file_names:
            edge_file_name = node_file_name.replace('nodes', 'edges')
            
            # load graph from file
            nodes_df = pd.read_csv(os.path.join(self.raw_dir, node_file_name))
            edges_df = pd.read_csv(os.path.join(self.raw_dir, edge_file_name))

            # drop nan rows
            nodes_df = nodes_df.dropna()

            # drop duplicate titles
            nodes_df = nodes_df.drop_duplicates(subset='title')

            graph = nx.DiGraph()

            node_mapping = {}
            # add nodes from dataframe
            print(f"Loading nodes into graph from {node_file_name}")
            for idx, node_row in tqdm.tqdm(nodes_df.iterrows(), total=len(nodes_df)):
                node_mapping[node_row['id']] = idx

                graph.add_node(idx, node_text=node_row['title'])

            # add edges from dataframe
            print(f"Loading edges into graph from {edge_file_name}")
            for idx, edge_row in tqdm.tqdm(edges_df.iterrows(), total=len(edges_df)):
                if edge_row['old_id'] not in node_mapping or edge_row['new_id'] not in node_mapping:
                    continue

                old_id = node_mapping[edge_row['old_id']]
                new_id = node_mapping[edge_row['new_id']]

                if graph.has_edge(old_id, new_id):
                    graph[old_id][new_id]['entities'] += 1
                else:
                    graph.add_edge(old_id, new_id, entities=1)

        

            # process network into torch compat shape
            print('Create context/target pairs from graph')
            for node, node_data in tqdm.tqdm(graph.nodes(data=True), total=graph.number_of_nodes()):
                if graph.in_degree(node) == 0:
                    continue

                predecessors = get_top_n_valid_predecessors(graph, node, 1)
                if len(predecessors) == 0:
                    continue

                context_node = predecessors[0]

                data = {}

                node_text = node_data['node_text']

                bert_tokens = bert_tokenizer(node_text, padding='max_length', truncation=True, max_length=MAX_TOKENS)
                lm_tokens = lm_tokenizer(node_text, padding='max_length', truncation=True, max_length=MAX_TOKENS)

                data['target_input_ids'] = torch.tensor(bert_tokens.input_ids, dtype=torch.long)
                data['target_input_attention_mask'] = torch.tensor(bert_tokens.attention_mask, dtype=torch.long)

                data['target_output_ids'] = torch.tensor(lm_tokens.input_ids, dtype=torch.long)
                data['target_output_attention_mask'] = torch.tensor(lm_tokens.attention_mask, dtype=torch.long)

                context_text = graph.nodes[context_node]['node_text']

                context_bert_tokens = bert_tokenizer(context_text, padding='max_length', truncation=True, max_length=MAX_TOKENS)
                
                data['conditioner_input_ids'] = torch.tensor(context_bert_tokens.input_ids)
                data['conditioner_attention_mask'] = torch.tensor(context_bert_tokens.attention_mask)

                context_text_with_stop = context_text + '. '
                context_with_stop_lm_tokens = lm_tokenizer(context_text_with_stop, padding='max_length', truncation=True, max_length=MAX_TOKENS)

                data['decoder_input_ids'] = torch.tensor(context_with_stop_lm_tokens.input_ids)
                data['decoder_attention_mask'] = torch.tensor(context_with_stop_lm_tokens.attention_mask)

                joined_text = context_text_with_stop + node_text
                joined_lm_tokens = lm_tokenizer(joined_text, padding='max_length', truncation=True, max_length=MAX_TOKENS)

                data['joined_input_ids'] = torch.tensor(joined_lm_tokens.input_ids)
                data['joined_attention_mask'] = torch.tensor(joined_lm_tokens.attention_mask)

                ancestors = get_n_gen_ancestors(graph, node, NUM_GENERATIONS, MAX_TOP_PREDECESSORS)
                graph_context = nx.induced_subgraph(graph, ancestors).copy()

                num_nodes = len(graph_context)
                node_token_ids = np.zeros((num_nodes, MAX_TOKENS))
                node_attention_mask = np.zeros((num_nodes, MAX_TOKENS))

                node_map = {}
                for node_idx, (context_node, context_node_data) in enumerate(graph_context.nodes(data=True)):
                    context_node_text = context_node_data['node_text']
                    context_node_bert_tokens = bert_tokenizer(context_node_text, padding='max_length', truncation=True, max_length=MAX_TOKENS)
                    node_token_ids[node_idx, :] = context_node_bert_tokens.input_ids
                    node_attention_mask[node_idx, :] = context_node_bert_tokens.attention_mask

                    node_map[context_node] = node_idx

                edges = []
                for (node_1, node_2) in graph_context.edges():
                    edges.append([node_map[node_1], node_map[node_2]])
                
                graph_context = torch_geometric.data.Data(
                    input_ids = torch.tensor(node_token_ids, dtype=torch.long),
                    attention_mask = torch.tensor(node_attention_mask, dtype=torch.long),
                    edge_index = torch.tensor(edges, dtype=torch.long).T
                )
                graph_context.num_nodes = num_nodes

                graph_data_list.append(graph_context)
                data_list.append(data)


        data_save_path = os.path.join(self.processed_dir, 'data.pt')
        graph_data_save_path = os.path.join(self.processed_dir, 'graph_data.pt')

        torch.save(data_list, data_save_path)
        data, slices = self.collate(graph_data_list)
        torch.save((data, slices), graph_data_save_path)


def load_and_preprocess_dataset(model, dataset_name, lm_name):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', dataset_name)
    graph_context = model == 'gcvae'

    dataset = NewsDataset(path, dataset_name, graph_context=graph_context, lm_name=lm_name)

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    return train_dataset, val_dataset, test_dataset