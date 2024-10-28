import math
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.distributions import MultivariateNormal, TransformedDistribution, SigmoidTransform, Normal
import torch.nn.functional as F


class EdgeMLP(nn.Module):
    def __init__(self, in_dim=10, hidden_dim=50, num_hidden_layers=3, out_dim=1, activation=nn.GELU(), output_activation=nn.Sigmoid()):
        super(EdgeMLP, self).__init__()
        layers = []

        # Input layer
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(activation)

        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation)

        # Output layer
        layers.append(nn.Linear(hidden_dim, out_dim))
        if output_activation is not None:
            layers.append(output_activation)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x, edge_index):
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Edge indices of shape [2, num_edges]

        # Extract source and target node indices
        src_nodes = edge_index[0]  # Source node indices
        tgt_nodes = edge_index[1]  # Target node indices

        # Gather the features for the source and target nodes
        x_src = x[src_nodes]  # Features of source nodes [num_edges, in_channels]
        x_tgt = x[tgt_nodes]  # Features of target nodes [num_edges, in_channels]

        # Concatenate the source and target features
        edge_features = torch.cat([x_src, x_tgt], dim=-1)  # [num_edges, 2 * in_channels]

        # Pass the concatenated features through the MLP
        out = self.mlp(edge_features)  # [num_edges, out_dim]

        return out  # Output features for each edge


class GraphTransformer(nn.Module):

class GraphGenerator(nn.Module):
    def __init__(self):
        super(GraphGenerator, self).__init__()
        self.gen = EdgeMLP()
        self.CnH = EdgeMLP()

        
    def evaluate_graph(self, token, edge_index):
        return self.VoQ.forward(token, edge_index)

    def generate_graph(self, token, edge_index):
        return self.CnH.forward(token, edge_index)
    

