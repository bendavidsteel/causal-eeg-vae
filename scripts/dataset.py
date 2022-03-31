
import os

import networkx as nx
import ntlk
import pandas as pd
import torch
import torch_geometric
import transformers

NUM_SENTENCES = 5
MAX_PREDECESSORS = 3

def get_top_n_predecessors(graph, node, n):
    predecessors = graph.predecessors(node)
    predec_weights = []
    for predecessor in predecessors:
        predec_weights.append((predecessor, graph[predecessor][node]['entities']))

    sorted_predecessors = [predec[0] for predec in sorted(predec_weights, key=lambda x: x[1], reverse=True)]

    return sorted_predecessors[:n]

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

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f"{self.dataset_name}_entity_dag_{data_type}.csv" for data_type in ['nodes', 'edges']]

    @property
    def processed_file_names(self):
        return ['news_entity_dag.pt']

    def process(self):

        nodes_df = pd.read_csv(os.path.join(self.raw_dir, f"{self.dataset_name}_entity_dag_nodes.csv"))
        edges_df = pd.read_csv(os.path.join(self.raw_dir, f"{self.dataset_name}_entity_dag_edges.csv"))

        graph = nx.DiGraph()

        roberta_tokenizer = transformers.RobertaTokenizerFast.from_pretrained("roberta-base")

        node_idx = 0
        node_mapping = {}
        for node_row in nodes_df.iterrows():
            node_mapping[node_row['id']] = node_idx

            first_sentences = ' '.join(nltk.tokenize.sent_tokenize(node_row['text'])[:NUM_SENTENCES - 1])
            node_text = node_row['title'] + '. ' + first_sentences
            node_token_ids = roberta_tokenizer.encode(node_text)
            graph.add_node(node_idx, token_ids=node_token_ids)

            node_idx += 1

        for edge_row in edges_df.iterrows():
            old_id = node_mapping[edge_row['old_id']]
            new_id = node_mapping[edge_row['new_id']]

            if graph.has_edge(old_id, new_id):
                graph[old_id][new_id]['entities'] += 1
            else:
                graph.add_edge(old_id, new_id, entities=1)

        for node in graph.nodes():
            
            if self.graph_context:
                context = 
            else:
                context_node = get_top_n_predecessors(graph, node, 1)[0]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def load_and_preprocess_dataset(dataset_name, device):
    transform = torch_geometric.transforms.Compose([
        torch_geometric.transforms.ToDevice(device),
    ])
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', dataset_name)
    dataset = NewsDataset(path, dataset_name, transform=transform)

    return dataset[0]