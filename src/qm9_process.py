import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Distance
from sklearn.preprocessing import StandardScaler
import os

class QM9Dataset:
    def __init__(self, root, batch_size=32):
        self.root = root
        self.batch_size = batch_size

        self.dataset = QM9(self.root, pre_transform=Distance())
        print(f"Number of data instances: {len(self.dataset)}")


    def process_dataset(self, target=4, verbosity=0):
        # Extract either HOMO-LUMO gap (target index 4) or Dipole moment (target index 0)
        processed_dataset = []

        for data in self.dataset:
            gap = data.y[:, target].unsqueeze(0)  # Extract only the target data
            data.y = gap

            # Set num_node_features and num_edge_features
            data.num_node_features = data.x.size(1)  # Number of features per node
            data.num_edge_features = data.edge_attr.size(1)  # Number of features per edge

            # Handle edge_weight
            if data.edge_weight is None:
                data.edge_weight = torch.ones_like(data.edge_index[0])  # Set edge weights to 1.0 if not provided

            processed_dataset.append(data)

        self.dataset = processed_dataset

        if verbosity>=1: 
            if target == 4:
                print(f"First 10 values of HOMO-LUMO gap values:")
            elif target == 0:
                print(f"First 10 values of dipole moment values:")
            for i in range(10):
                print(self.dataset[i].y)


    def get_dataloaders(self, split_ratio=(0.8, 0.1, 0.1)):
        train_size = int(split_ratio[0] * len(self.dataset))
        val_size = int(split_ratio[1] * len(self.dataset))
        test_size = len(self.dataset) - train_size - val_size

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(self.dataset, [train_size, val_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader


    def print_instance(self, numMolPrint=1):
        # Print the features of the first molecule from the processed data
        for i, data in enumerate(self.dataset):
            if i < numMolPrint:
                print(f"\tProcessed QM9 Data ({i}-th Instance):")
                print(f"\tNumber of nodes (atoms): {data.num_nodes}")
                print(f"\tNumber of edges (bonds): {data.num_edges}")
                print(f"\tNode features (x): {data.x}")
                print(f"\tEdge indices (edge_index): {data.edge_index}")
                print(f"\tEdge weights (edge_weight): {data.edge_weight if data.edge_weight is not None else 'None'}")
                print(f"\tEdge attributes (edge_attr): {data.edge_attr}")
                print(f"\tAtomic positions (pos): {data.pos}") 
                print(f"\tAtomic numbers (z): {data.z}") 
                print(f"\tSMILES representation (smiles): {data.smiles}") 
                print(f"\tTarget values (y): {data.y}")  # This could be properties like HOMO-LUMO gap
                print("=" * 80)

