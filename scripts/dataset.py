
import os

import networkx as nx
import ntlk
import numpy as np
import pandas as pd
import torch
import torch_geometric
import transformers

NUM_SENTENCES = 5
MAX_TOKENS = 200
MAX_TOP_PREDECESSORS = 3
NUM_GENERATIONS = 3

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

        self._process()

        context_save_path = os.path.join(self.processed_dir, 'contexts.pt')
        target_save_path = os.path.join(self.processed_dir, 'targets.pt')

        self.targets = torch.load(target_save_path)
        self.contexts = torch.save(context_save_path)

    def get(self, idx: int):
        if self.len() == 1:
            return copy.copy(self.data)

        if not hasattr(self, '_data_list') or self._data_list is None:
            self._data_list = self.len() * [None]
        elif self._data_list[idx] is not None:
            return copy.copy(self._data_list[idx])

        data = separate(
            cls=self.data.__class__,
            batch=self.data,
            idx=idx,
            slice_dict=self.slices,
            decrement=False,
        )

        self._data_list[idx] = copy.copy(data)

        return data

    @property
    def raw_file_names(self):
        return [f"{self.dataset_name}_entity_dag_{data_type}.csv" for data_type in ['nodes', 'edges']]

    @property
    def processed_file_names(self):
        return ['news_entity_dag.pt']

    def _process(self):

        # load graph from file
        nodes_df = pd.read_csv(os.path.join(self.raw_dir, f"{self.dataset_name}_entity_dag_nodes.csv"))
        edges_df = pd.read_csv(os.path.join(self.raw_dir, f"{self.dataset_name}_entity_dag_edges.csv"))

        graph = nx.DiGraph()

        roberta_tokenizer = transformers.RobertaTokenizerFast.from_pretrained("roberta-base")

        node_idx = 0
        node_mapping = {}
        # add nodes from dataframe
        for node_row in nodes_df.iterrows():
            node_mapping[node_row['id']] = node_idx

            first_sentences = ' '.join(nltk.tokenize.sent_tokenize(node_row['text'])[:NUM_SENTENCES - 1])
            node_text = node_row['title'] + '. ' + first_sentences
            node_token_ids = roberta_tokenizer.encode(node_text, padding='max_length', max_length=MAX_TOKENS)

            graph.add_node(node_idx, token_ids=node_token_ids)

            node_idx += 1

        # add edges from dataframe
        for edge_row in edges_df.iterrows():
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
        for node in graph.nodes():
            if graph.in_degree(node) == 0:
                continue

            target = torch.tensor(graph[node]['token_ids'])
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


def load_and_preprocess_dataset(dataset_name, device):
    transform = torch_geometric.transforms.Compose([
        torch_geometric.transforms.ToDevice(device),
    ])
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', dataset_name)
    dataset = NewsDataset(path, dataset_name, transform=transform)

    return dataset[0]