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

        context_save_path = os.path.join(self.processed_dir, 'contexts.pt')
        target_save_path = os.path.join(self.processed_dir, 'targets.pt')

        self.targets = torch.load(target_save_path)
        self.contexts = torch.load(context_save_path)

    def len(self):
        return len(self.targets)

    def get(self, idx: int):

        if self.graph_context:
            graph_data, graph_slices = self.contexts
            
            if not hasattr(self, '_data_list') or self._data_list is None:
                self._data_list = self.len() * [None]
            elif self._data_list[idx] is not None:
                return copy.copy(self._data_list[idx]), self.targets[idx]

            graph_context = torch_geometric.data.separate.separate(
                cls=graph_data.__class__,
                batch=graph_data,
                idx=idx,
                slice_dict=graph_slices,
                decrement=False,
            )

            self._data_list[idx] = copy.copy(graph_context)

            return graph_context, self.targets[idx]
        else:
            return self.contexts[idx], self.targets[idx]

    @property
    def raw_file_names(self):
        return [f"{self.dataset_name}_entity_dag_{data_type}.csv" for data_type in ['nodes', 'edges']]

    @property
    def processed_file_names(self):
        return ['contexts.pt', 'targets.pt']

    def process(self):

        # load graph from file
        nodes_df = pd.read_csv(os.path.join(self.raw_dir, f"{self.dataset_name}_entity_dag_nodes.csv"))
        edges_df = pd.read_csv(os.path.join(self.raw_dir, f"{self.dataset_name}_entity_dag_edges.csv"))

        # drop nan rows
        nodes_df = nodes_df.dropna()

        graph = nx.DiGraph()

        roberta_tokenizer = transformers.RobertaTokenizerFast.from_pretrained("roberta-base")

        node_mapping = {}
        # add nodes from dataframe
        print('Loading nodes into graph')
        for idx, node_row in tqdm.tqdm(nodes_df.iterrows()):
            node_mapping[node_row['id']] = idx

            first_sentences = ' '.join(nltk.tokenize.sent_tokenize(node_row['text'])[:NUM_SENTENCES - 1])
            node_text = node_row['title'] + '. ' + first_sentences
            node_token_ids = roberta_tokenizer.encode(node_text, padding='max_length', truncation='longest_first', max_length=MAX_TOKENS)

            graph.add_node(idx, token_ids=node_token_ids)

        # add edges from dataframe
        print('Loading edges into graph')
        for idx, edge_row in tqdm.tqdm(edges_df.iterrows()):
            if edge_row['old_id'] not in node_mapping or edge_row['new_id'] not in node_mapping:
                continue

            old_id = node_mapping[edge_row['old_id']]
            new_id = node_mapping[edge_row['new_id']]

            if graph.has_edge(old_id, new_id):
                graph[old_id][new_id]['entities'] += 1
            else:
                graph.add_edge(old_id, new_id, entities=1)

        contexts = []
        targets = []

        # process network into torch compat shape
        print('Create context/target pairs from graph')
        for node, token_ids in tqdm.tqdm(graph.nodes(data='token_ids')):
            if graph.in_degree(node) == 0:
                continue

            target = torch.tensor(token_ids, dtype=torch.long)
            targets.append(target)

            if self.graph_context:
                ancestors = get_n_gen_ancestors(graph, node, NUM_GENERATIONS, MAX_TOP_PREDECESSORS)
                graph_context = nx.induced_subgraph(graph, ancestors).copy()

                num_nodes = len(graph_context)
                node_token_ids = np.zeros((num_nodes, MAX_TOKENS))

                node_map = {}
                for node_idx, (context_node, context_token_ids) in enumerate(graph_context.nodes(data='token_ids')):
                    node_token_ids[node_idx, :] = context_token_ids

                    node_map[context_node] = node_idx

                edges = []
                for (node_1, node_2) in graph_context.edges():
                    edges.append([node_map[node_1], node_map[node_2]])
                
                context = torch_geometric.data.Data(
                    x = torch.tensor(node_token_ids, dtype=torch.long),
                    edge_index = torch.tensor(edges, dtype=torch.long).T
                )

            else:
                context_node = get_top_n_predecessors(graph, node, 1)[0]
                context = torch.tensor(graph[context_node]['token_ids'])

            contexts.append(context)

        context_save_path = os.path.join(self.processed_dir, 'contexts.pt')
        target_save_path = os.path.join(self.processed_dir, 'targets.pt')

        torch.save(targets, target_save_path)
        if self.graph_context:
            data, slices = self.collate(contexts)
            torch.save((data, slices), context_save_path)
        else:
            torch.save(contexts, context_save_path)


def load_and_preprocess_dataset(model, dataset_name):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', dataset_name)
    graph_context = model == 'gcvae'

    dataset = NewsDataset(path, dataset_name, graph_context=graph_context)

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    train_dataloader = torch_geometric.loader.DataLoader(train_dataset, batch_size=32)
    val_dataloader = torch_geometric.loader.DataLoader(val_dataset, batch_size=32)
    test_dataloader = torch_geometric.loader.DataLoader(test_dataset, batch_size=32)

    return train_dataloader, val_dataloader, test_dataloader