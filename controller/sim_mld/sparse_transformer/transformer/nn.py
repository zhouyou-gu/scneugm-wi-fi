import torch
import torch.nn as nn
import torch.nn.functional as F
    
from torch_geometric.nn import TransformerConv
from torch_geometric.data import Data, Batch

class GraphTransformer(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=5, heads=5, num_layers=3):
        super(GraphTransformer, self).__init__()
        self.i_lin = nn.Sequential(
                nn.Linear(input_dim, hidden_dim*heads),
        )
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(TransformerConv(hidden_dim*heads, hidden_dim, heads=heads))
        self.o_lin = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
        )
    def forward(self, x, edge_index):
        x = self.i_lin(x)
        for conv in self.convs:
            x = conv(x, edge_index)
        x = self.o_lin(x)

        return x

    def process_pairs(self, node_features_list, index_pairs):
        graphs = []

        # Create individual graph data for each specified pair
        for i, (idx1, idx2) in enumerate(index_pairs):
            # Extract node features for the given index pair
            x_pair = torch.stack([node_features_list[idx1], node_features_list[idx2]])  # Shape: [2, input_dim]
            edge_index_pair = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)  # Bidirectional edge between nodes 0 and 1
            graphs.append(Data(x=x_pair, edge_index=edge_index_pair))

        # Batch all pair-wise graphs
        batch = Batch.from_data_list(graphs)

        # Forward pass through the transformer
        x = self.forward(batch.x, batch.edge_index)

        # Reshape the output for each pair
        return x.view(len(index_pairs), 2, -1)