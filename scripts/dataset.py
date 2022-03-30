
import os

import networkx as nx
import ntlk
import pandas as pd
import torch
import torch_geometric

NUM_SENTENCES = 5

class NewsDataset(torch_geometric.data.InMemoryDataset):
    def __init__(self, root, dataset_name='main', transform=None, pre_transform=None, pre_filter=None):

        self.dataset_name = dataset_name

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f"{self.dataset_name}_entity_dag_{data_type}.csv" for data_type in ['nodes', 'edges']]

    @property
    def processed_file_names(self):
        return ['news_entity_dag.pt']
        ...

    def process(self):

        nodes_df = pd.read_csv(os.path.join(self.raw_dir, f"{self.dataset_name}_entity_dag_nodes.csv"))
        edges_df = pd.read_csv(os.path.join(self.raw_dir, f"{self.dataset_name}_entity_dag_edges.csv"))

        graph = nx.DiGraph()

        node_idx = 0
        node_mapping = {}
        for node_row in nodes_df.iterrows():
            node_mapping[node_row['id']] = node_idx

            first_sentences = ' '.join(nltk.tokenize.sent_tokenize(node_row['text'])[:NUM_SENTENCES - 1])
            graph.add_node(node_idx, text=node_row['title'] + '. ' + first_sentences)

            node_idx += 1

        for edge_row in edges_df.iterrows():
            old_id = node_mapping[edge_row['old_id']]
            new_id = node_mapping[edge_row['new_id']]
            if graph.has_edge(old_id, new_id):
            else:
                graph.add_edge(old_id, new_id)

        for node in graph.nodes(data='text'):


        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def load_and_preprocess_dataset(device):
    transform = torch_geometric.transforms.Compose([
        torch_geometric.transforms.ToDevice(device),
        torch_geometric.transforms.RandomNodeSplit(num_val=0.05, num_test=0.1),
    ])
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'Planetoid')
    dataset = Planetoid(path, args.dataset, transform=transform)

    return dataset[0]