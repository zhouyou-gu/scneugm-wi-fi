import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv.transformer_conv import *
class SigmoidTransformerConv(TransformerConv):
    def __init__(self, in_channels, out_channels, heads=1, **kwargs):
        super(SigmoidTransformerConv, self).__init__(in_channels, out_channels, heads, **kwargs)

    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads,
                                                    self.out_channels)
            key_j = key_j + edge_attr

        # Compute attention scores without softmax
        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)

        # Apply sigmoid to ensure alpha is between 0 and 1
        alpha = torch.sigmoid(alpha)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value_j
        if edge_attr is not None:
            out = out + edge_attr

        out = out * alpha.view(-1, self.heads, 1)
        return out
    
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
                nn.Linear(hidden_dim*heads, 1),
                nn.Sigmoid()
        )
    def forward(self, x, edge_index):
        x = self.i_lin(x)
        for conv in self.convs:
            x = conv(x, edge_index)
        x = self.o_lin(x)

        return x