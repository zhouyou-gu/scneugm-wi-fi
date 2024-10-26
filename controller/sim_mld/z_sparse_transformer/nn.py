import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax

import torch_geometric as pyg

from torch_geometric.nn import GCNConv, SAGEConv, TransformerConv
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pack_sequence, pad_sequence

HIDDEN_DIM_MULTIPLIER = 10
N_FIELD_KQMV = 4
N_FIELD_KQ = 2

def init_weights(m,gain=1.):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight,gain=gain)
        torch.nn.init.zeros_(m.bias)


class attention_layer_KQMV(nn.Module):
    def __init__(self, token_dim, embed_dim= 5, heads= 5, concat=True):
        nn.Module.__init__(self)
        self.token_dim = token_dim
        self.embed_dim = embed_dim
        self.heads = heads
        self.concat = concat

        self.query = nn.Linear(token_dim, embed_dim * heads, bias=False)
        self.key = nn.Linear(token_dim, embed_dim * heads, bias=False)
        self.message = nn.Linear(token_dim, embed_dim * heads, bias=False)
        self.value = nn.Linear(token_dim, embed_dim * heads, bias=False)

        self.out_proj = nn.Linear(embed_dim*heads*2, token_dim, bias=False)

    def forward(self, x, edge_index):
        # Get the query, key, and value embeddings
        Q = self.query(x)       # (N, out_channels * heads)
        K = self.key(x)         # (N, out_channels * heads)
        M = self.message(x)     # (N, out_channels * heads)
        V = self.value(x)       # (N, out_channels * heads)

        # Reshape to (N, heads, token_dim)
        Q = Q.view(-1, self.heads, self.embed_dim)
        K = K.view(-1, self.heads, self.embed_dim)
        M = M.view(-1, self.heads, self.embed_dim)
        V = V.view(-1, self.heads, self.embed_dim)

        # Compute attention scores only for the edges in edge_index
        row, col = edge_index  # edge_index shape is [2, E]
        attention_scores = torch.einsum('ehc, ehc -> e', Q[row], K[col]) / (self.embed_dim ** 0.5)  # (E,)

        # Normalize the attention scores using softmax
        attention_probs = softmax(attention_scores, row)  # Softmax over the neighbors

        # Compute the weighted sum of values
        out = torch.einsum('e, ehc -> ehc', attention_probs, M[col])

        # Aggregate the weighted values by summing them up for each node
        out = torch.zeros_like(M).scatter_add_(0, row.unsqueeze(-1).unsqueeze(-1).expand_as(out), out)

        # Concatenate self-value with aggregated neighbor values
        out = torch.cat([V, out], dim=-1)  # Concatenate along the feature dimension

        # Concatenate or average heads, depending on the configuration
        if self.concat:
            out = out.view(-1, self.embed_dim * self.heads * 2)
        else:
            out = out.mean(dim=1)

        # Apply final linear transformation
        out = self.out_proj(out)

        return out

class network_performance_transformer(nn.Module):
    
    def __init__(self, token_dim=10, head_dim=5, num_heads=5, ot_dim=1, num_layers=3):
        nn.Module.__init__(self)
        self.attention_layers = nn.ModuleList([
            attention_layer_KQMV(token_dim, head_dim, heads=num_heads) for _ in range(num_layers)
        ])
        self.ot_node_emb = nn.Sequential(
                        nn.Linear(token_dim, token_dim*HIDDEN_DIM_MULTIPLIER),
                        nn.ReLU(),
                        nn.Linear(token_dim*HIDDEN_DIM_MULTIPLIER, token_dim*HIDDEN_DIM_MULTIPLIER),
                        nn.ReLU(),
                        nn.Linear(token_dim*HIDDEN_DIM_MULTIPLIER, ot_dim),
        )
    def forward(self, x, edge_index):
        for attention_layer in self.attention_layers:
            x = attention_layer(x, edge_index)
            x = F.relu(x)
        x = self.ot_node_emb(x)
        return x
class sparser_transformer(nn.Module):
    def __init__(self, token_dim=10, embed_dim=25, num_heads=5, ot_dim = 20, num_layers=3):
        nn.Module.__init__(self)
        assert num_layers > 1
        self.in_node_emb = nn.Sequential(
                nn.Linear(in_dim, in_dim * HIDDEN_DIM_MULTIPLIER),
                nn.ReLU(),
                nn.Linear(in_dim * HIDDEN_DIM_MULTIPLIER, in_dim * HIDDEN_DIM_MULTIPLIER),
                nn.ReLU(),
                nn.Linear(in_dim * HIDDEN_DIM_MULTIPLIER, embed_dim),
                nn.ReLU(),
        )
        self.transformer_layers = nn.ModuleList([
            TransformerConv(embed_dim, embed_dim // num_heads, heads=num_heads)
            for _ in range(num_layers)
        ])
        self.ot_node_emb = nn.Sequential(
                        nn.Linear(embed_dim, embed_dim),
                        nn.ReLU(),
                        nn.Linear(embed_dim, embed_dim),
                        nn.ReLU(),
                        nn.Linear(embed_dim, ot_dim),
        )

    def forward(self, x, edge_index):
        x = self.in_node_emb.forward(x)
        for i, transformer in enumerate(self.transformer_layers):
            x = transformer(x, edge_index)

        x = self.ot_node_emb(x)

        tokens = F.normalize(x, p=2, dim=1)

        return tokens
    
    
if __name__ == "__main__":
    from torch.nn.utils.rnn import pack_sequence
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([4, 5])
    c = torch.tensor([6])
    sq = pack_sequence([b, a, c],enforce_sorted=False)
    print(sq)