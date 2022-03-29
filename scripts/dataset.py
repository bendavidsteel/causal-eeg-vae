
import os

def load_and_preprocess_dataset(device):
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                        split_labels=True, add_negative_train_samples=False),
    ])
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'Planetoid')
    dataset = Planetoid(path, args.dataset, transform=transform)

    return dataset[0]