import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

HIDDEN_DIM_MULTIPLIER = 5

class predictor(nn.Module):
    def __init__(self, input_dim=5*2):
        super(predictor, self).__init__()
        hidden_dim = input_dim*HIDDEN_DIM_MULTIPLIER

        # Input layer
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
    def forward(self, x, edge_index):
        # Extract source and target node indices
        src_nodes = edge_index[0]
        tgt_nodes = edge_index[1]

        # Gather the features for the source and target nodes
        x_src = x[src_nodes]  # [num_edges, in_dim_node + token_dim]
        x_tgt = x[tgt_nodes]  # [num_edges, in_dim_node + token_dim]

        edge_features = torch.cat([x_src, x_tgt], dim=-1) 

        return self.mlp(edge_features)



 