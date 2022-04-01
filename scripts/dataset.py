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

NUM_SENTENCES = 5
MAX_TOKENS = 200
MAX_TOP_PREDECESSORS = 3
NUM_GENERATIONS = 3

ContextTargetData = collections.namedtuple('ContextTargetData', ['target', 'context'])

def get_top_n_predecessors(graph, node, n):
    predecessors = graph.predecessors(node)
    predec_weights = []
    for predecessor in predecessors:
        predec_weights.append((predecessor, graph[predecessor][node]['entities']))

    sorted_predecessors = [predec[0] for predec in sorted(predec_weights, key=lambda x: x[1], reverse=True)]

    return sorted_predecessors[:min(n, len(predecessors))]

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
        self.contexts = torch.save(context_save_path)

    def len(self):
        return len(self.targets)

    def get(self, idx: int):

        if self.graph_context:
            graph_data, graph_slices = self.contexts
            target = self.targets[idx]

            if not hasattr(self, '_data_list') or self._data_list is None:
                self._data_list = self.len() * [None]
            elif self._data_list[idx] is not None:
                return ContextTargetData(target, copy.copy(self._data_list[idx]))

            graph_context = torch_geometric.data.separate(
                cls=graph_data.__class__,
                batch=graph_data,
                idx=idx,
                slice_dict=graph_slices,
                decrement=False,
            )

            self._data_list[idx] = copy.copy(graph_context)

            return ContextTargetData(target, graph_context)
        else:
            return ContextTargetData(self.targets[idx], self.contexts[idx])

    @property
    def raw_file_names(self):
        return [f"{self.dataset_name}_entity_dag_{data_type}.csv" for data_type in ['nodes', 'edges']]

    @property
    def processed_file_names(self):
        return ['news_entity_dag.pt']

    def process(self):

        # load graph from file
        nodes_df = pd.read_csv(os.path.join(self.raw_dir, f"{self.dataset_name}_entity_dag_nodes.csv"))
        edges_df = pd.read_csv(os.path.join(self.raw_dir, f"{self.dataset_name}_entity_dag_edges.csv"))

        graph = nx.DiGraph()

        roberta_tokenizer = transformers.RobertaTokenizerFast.from_pretrained("roberta-base")

        node_mapping = {}
        # add nodes from dataframe
        print('Loading nodes into graph')
        for idx, node_row in tqdm.tqdm(nodes_df.iterrows()):
            node_mapping[node_row['id']] = idx

            if node_row['text'] is not str or node_row['title'] is not str:
                continue

            first_sentences = ' '.join(nltk.tokenize.sent_tokenize(node_row['text'])[:NUM_SENTENCES - 1])
            node_text = node_row['title'] + '. ' + first_sentences
            node_token_ids = roberta_tokenizer.encode(node_text, padding='max_length', max_length=MAX_TOKENS)

            graph.add_node(idx, token_ids=node_token_ids)

        # add edges from dataframe
        print('Loading edges into graph')
        for idx, edge_row in tqdm.tqdm(edges_df.iterrows()):
            old_id = node_mapping[edge_row['old_id']]
            new_id = node_mapping[edge_row['new_id']]

            if graph.has_edge(old_id, new_id):
                graph[old_id][new_id]['entities'] += 1
            else:
                graph.add_edge(old_id, new_id, entities=1)

        contexts = []
        targets = []

        # process network into torch compat shape
        num_data_points = 0
        print('Create context/target pairs from graph')
        for node, token_ids in tqdm.tqdm(graph.nodes(data='token_ids')):
            if graph.in_degree(node) == 0:
                continue

            target = torch.tensor(token_ids)
            targets.append(target)

            if self.graph_context:
                ancestors = get_n_gen_ancestors(graph, node, NUM_GENERATIONS, MAX_TOP_PREDECESSORS)
                graph_context = nx.induced_subgraph(graph, ancestors).copy()

                num_nodes = len(graph_context)
                node_token_ids = np.zeros((num_nodes, MAX_TOKENS))

                node_map = {}
                node_cnt = 0
                for node in graph_context.nodes():
                    node_token_ids[node_cnt, :] = graph_context[node]['token_ids']

                    node_map[node] = node_cnt
                    node_cnt += 1

                edges = []
                for (node_1, node_2) in graph_context.edges():
                    edges.append([node_map[node_1], node_map[node_2]])
                
                context = torch_geometric.data.Data(
                    x = torch.tensor(node_token_ids),
                    edge_index = torch.tensor(edges, dtype=torch.long).T
                )

            else:
                context_node = get_top_n_predecessors(graph, node, 1)[0]
                context = torch.tensor(graph[context_node]['token_ids'])

            contexts.append(context)
            num_data_points += 1

        self.num_data_points = num_data_points

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

    train_dataloader = torch.utils.data.DataLoader(train_dataset)
    val_dataloader = torch.utils.data.DataLoader(val_dataset)
    test_dataloader = torch.utils.data.DataLoader(test_dataset)

    return train_dataloader, val_dataloader, test_dataloader