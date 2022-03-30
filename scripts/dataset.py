
import os

import torch
import torch_geometric

class NewsDataset(torch_geometric.data.InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['nodes.csv', 'edges.csv']

    @property
    def processed_file_names(self):
        return ['news_entity_dag.pt']
        ...

    def process(self):
        # Read data into huge `Data` list.
        data_list = [...]

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