import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv.transformer_conv import *
from torch_geometric.nn import GCNConv, SimpleConv, MessagePassing, TransformerConv
from torch_geometric.utils import add_self_loops

COLORING_RELATED_EDGE_DIM =2
HIDDEN_DIM_MULTIPLIER = 5

class MessagePassingNNWithEdge(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim=1, hidden_channels=10):
        super(MessagePassingNNWithEdge, self).__init__(aggr='add')
        
        # MLP that takes node features and edge attributes
        self.mlp = nn.Linear(2 * in_channels + edge_dim, out_channels,bias=False)
        self.update_mlp = nn.Linear(out_channels, out_channels,bias=False)
    
    def forward(self, x, edge_index, edge_attr):
        # Add self-loops to the adjacency matrix
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Create self-loop edge attributes (e.g., zeros)
        # num_new_edges = x.size(0)  # One self-loop per node
        # self_loop_attr = torch.zeros((num_new_edges, edge_attr.size(1)), device=edge_attr.device)
        
        # # Concatenate original edge attributes with self-loop attributes
        # edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)
        
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i, x_j, edge_attr):
        # Concatenate target node, source node, and edge attributes
        combined = torch.cat([x_i, x_j, edge_attr], dim=1)

        messages = self.mlp(combined)
        return messages
    
    def update(self, aggr_out):
        updated = self.update_mlp(aggr_out)
        return updated

class EdgeMLP(nn.Module):
    def __init__(self, in_dim_node=6, in_dim_edge=1, hidden_dim=10, num_hidden_layers=1, out_dim=2, activation=nn.ReLU(), output_activation=None, symmetric=True):
        super(EdgeMLP, self).__init__()
        self.symmetric = symmetric
        layers = []

        # Input layer takes concatenated node and edge features
        layers.append(nn.Linear(2 * in_dim_node + in_dim_edge, hidden_dim))
        layers.append(activation)

        # Hidden layers process the combined features
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation)

        # Output layer produces the final edge output
        layers.append(nn.Linear(hidden_dim, out_dim))
        if output_activation is not None:
            layers.append(output_activation)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x, edge_index, edge_attr, edge_attr_T = None):
        """
        Forward pass for EdgeMLP.

        Args:
            x (Tensor): Node feature matrix of shape [num_nodes, in_dim_node + token_dim].
            edge_index (Tensor): Edge indices of shape [2, num_edges].
            edge_attr (Tensor): Edge feature matrix of shape [num_edges, in_dim_edge] or [num_edges].

        Returns:
            Tensor: Binary outputs for each edge of shape [num_edges, out_dim].
        """
        # Ensure edge_attr is 2D
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)  # Convert to [num_edges, 1]
        elif edge_attr.dim() != 2:
            raise ValueError(f"Expected edge_attr to have 1 or 2 dimensions, got {edge_attr.dim()}")
        
        if self.symmetric:
            assert edge_attr_T is not None, "symmetric mode need the edge_attr_T"
            # Ensure edge_attr_sym is 2D
            if edge_attr_T.dim() == 1:
                edge_attr_T = edge_attr_T.unsqueeze(-1)  # Convert to [num_edges, 1]
            elif edge_attr_T.dim() != 2:
                raise ValueError(f"Expected edge_attr_T to have 1 or 2 dimensions, got {edge_attr_T.dim()}")
            assert edge_attr_T.shape == edge_attr.shape, "edge_attr_T needs to be in the same shape as edge_attr"

        # Extract source and target node indices
        src_nodes = edge_index[0]
        tgt_nodes = edge_index[1]

        # Gather the features for the source and target nodes
        x_src = x[src_nodes]  # [num_edges, in_dim_node + token_dim]
        x_tgt = x[tgt_nodes]  # [num_edges, in_dim_node + token_dim]

        # Concatenate the source and target node features with edge features
        edge_features = torch.cat([x_src, x_tgt, edge_attr], dim=-1)  # [num_edges, 2 * (in_dim_node + token_dim) + in_dim_edge]

        # Pass the concatenated features through the MLP to get edge outputs
        out = self.mlp(edge_features)  # [num_edges, out_dim]
        
        if self.symmetric:
            edge_features = torch.cat([x_tgt, x_src, edge_attr_T], dim=-1)  # [num_edges, 2 * (in_dim_node + token_dim) + in_dim_edge]
            out += self.mlp(edge_features)  # [num_edges, out_dim]
            out = out/2.
        
        out = F.softmax(out,dim=1)  # [num_nodes, out_dim]

        out = out[:,-1].unsqueeze(-1)
        return out

# Not Used
# Custom TransformerConv that applies a mask to the attention scores based on edge attributes
class MaskedTransformerConv(TransformerConv):
    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        if self.lin_edge is not None:
            assert edge_attr is not None
            if edge_attr.size(1) < 3:
                raise ValueError("edge_attr must have at least 3 features for masking.")
            mask = edge_attr[:, -2] * (1-edge_attr[:, -1])
            edge_attr_main = edge_attr[:, :-2]
            edge_attr = self.lin_edge(edge_attr_main).view(-1, self.heads,
                                                      self.out_channels)
            key_j = key_j + edge_attr

        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        # alpha = torch.sigmoid(alpha)

        # alpha = softmax(alpha, index, ptr, size_i)
        if mask is not None:
            # Expand mask to match the number of heads
            mask = mask.unsqueeze(1).expand(-1, self.heads)  # Shape: [num_edges, heads]
            # Apply mask: set attention scores to -inf where mask is False
            alpha = alpha*mask
        
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value_j
        if edge_attr is not None:
            out = out + edge_attr

        out = out * alpha.view(-1, self.heads, 1)
        return out

class GCNEvaluator(nn.Module):
    def __init__(self, input_dim, edge_dim, num_layers, output_dim=2, hidden_dim=10, activation=nn.ReLU(), conv="GCNConv"):
        """
        Initializes the GCNEvaluator model using GCN layers with edge attributes.

        Args:
            input_dim (int): Dimension of the input node features.
            edge_dim (int): Dimension of the edge features.
            hidden_dim (int): Dimension of the hidden layers.
            num_layers (int): Number of GCN layers.
            output_dim (int): Dimension of the output layer.
            activation (nn.Module): Activation function.
        """
        super(GCNEvaluator, self).__init__()
        assert edge_dim>COLORING_RELATED_EDGE_DIM, "edge_dim needs >= 3"
        
        self.edge_weight_lin = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim,bias=False),
            activation,
            nn.Linear(hidden_dim, hidden_dim,bias=False),
            activation
        )

        self.i_lin = nn.Linear(input_dim, hidden_dim)
        
        self.convs = nn.ModuleList()
        
        for _ in range(num_layers):
            self.convs.append(activation)
            tmp = nn.ModuleList()
            for _ in range(hidden_dim):
                if conv == "GCNConv":
                    tmp.append(GCNConv(1, 1,add_self_loops=True,normalize=True))
                elif conv == "TransformerConv":
                    tmp.append(TransformerConv(in_channels=1,out_channels=1,edge_dim=1))
                elif conv == "MessagePassingNNWithEdge":
                    tmp.append(MessagePassingNNWithEdge(in_channels=1,out_channels=1,edge_dim=1))
                elif conv == "SimpleConv":
                    tmp.append(SimpleConv())
                else:
                    raise Exception("Undefined Conv in GCNEvaluator")
            self.convs.append(tmp)
            self.convs.append(nn.Linear(hidden_dim+hidden_dim, hidden_dim,bias=False))
        
        self.convs.append(activation)
        self.o_lin = nn.Linear(hidden_dim+hidden_dim, output_dim,bias=False)
        
    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass for GCNEvaluator.

        Args:
            x (Tensor): Node feature matrix of shape [num_nodes, input_dim].
            edge_index (Tensor): Edge indices of shape [2, num_edges].
            edge_attr (Tensor, optional): Edge feature matrix of shape [num_edges, edge_dim].

        Returns:
            Tensor: Output tensor for each node of shape [num_nodes, output_dim].
        """
        if edge_attr is not None:
            if edge_attr.size(1) < 3:
                raise ValueError("edge_attr must have at least 3 features for masking.")
            edge_weight = self.edge_weight_lin(edge_attr).squeeze(-1)  # <-- Transform edge_attr to scalar weights
        else:
            edge_weight = None

        x_ = self.i_lin(x)
        x = x_
        for layer in self.convs:
            if isinstance(layer, nn.ModuleList):
                x = torch.cat([e.forward(x[:,i].unsqueeze(-1), edge_index, edge_weight[:,i].unsqueeze(-1)) for i, e in enumerate(layer)], dim=1)
            elif isinstance(layer, nn.Linear):
                x = layer(torch.cat([x_,x],dim=1))
                x = x+x_
            else:
                x = layer(x)
        x = self.o_lin(torch.cat([x_,x],dim=1))
        x = F.softmax(x,dim=1)  # [num_nodes, 2]
        return x[:, 1].unsqueeze(-1)

class GraphGenerator(nn.Module):
    def __init__(self, node_dim=1, token_dim=5, edge_dim=1, token_enabled=False):
        """
        Initializes the GraphGenerator with specified dimensions for nodes, tokens, and edges.

        Args:
            node_dim (int): Dimension of the node features.
            token_dim (int): Dimension of the token features.
            edge_dim (int): Dimension of the edge features.
        """
        super(GraphGenerator, self).__init__()
        self.token_enabled = token_enabled
        if not self.token_enabled:
            token_dim = 0
            
        self.graph_generator = EdgeMLP(
            in_dim_node=node_dim + token_dim,  # Combined node and token dimensions
            in_dim_edge=edge_dim,
            out_dim=2
        )
        self.graph_evaluator_t = GCNEvaluator(
            input_dim=node_dim + token_dim,
            edge_dim=edge_dim + COLORING_RELATED_EDGE_DIM,  # Including the generated edge value and color collision
            num_layers=1
        )
        self.graph_evaluator_c = GCNEvaluator(
            input_dim=node_dim + token_dim,
            edge_dim=edge_dim + COLORING_RELATED_EDGE_DIM,  # Including the generated edge value and color collision
            num_layers=10
        )
    def generate_graph(self, x, token, edge_attr, edge_index, edge_attr_T = None):
        """
        Generates a binary value for each edge based on node and edge features.

        Args:
            x (Tensor): Node feature matrix of shape [num_nodes, in_dim_node].
            edge_index (Tensor): Edge indices of shape [2, num_edges].
            edge_attr (Tensor): Edge feature matrix of shape [num_edges, in_dim_edge].

        Returns:
            Tensor: Binary values for each edge of shape [num_edges, 1].
        """
        if self.token_enabled:
            # Ensure x and token have the same number of dimensions
            if token.dim() == 1:
                token = token.unsqueeze(-1)  # Convert to [num_nodes, 1]
            elif token.dim() != 2:
                raise ValueError(f"Expected token to have 1 or 2 dimensions, got {token.dim()}")

        if x.dim() == 1:
            x = x.unsqueeze(-1)  # Convert to [num_nodes, 1]
        elif x.dim() != 2:
            raise ValueError(f"Expected x to have 1 or 2 dimensions, got {x.dim()}")

        # Ensure edge_attr is 2D
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)  # Convert to [num_edges, 1]
        elif edge_attr.dim() != 2:
            raise ValueError(f"Expected edge_attr to have 1 or 2 dimensions, got {edge_attr.dim()}")
        
        
        # Concatenate node features with token features
        if self.token_enabled:
            x = torch.cat([x, token], dim=-1)
        
        # Use the EdgeMLP to generate probabilty for each edge
        edge_values = self.graph_generator(x, edge_index, edge_attr, edge_attr_T)  # [num_edges, 2]

        return edge_values

    def evaluate_graph(self, x, token, edge_value, edge_attr, edge_index):
        """
        Evaluates the graph to produce node-level outputs based on node and edge features.

        Args:
            x (Tensor): Node feature matrix of shape [num_nodes, in_dim_node].
            token (Tensor): Node token features of shape [num_nodes, token_dim].
            edge_value (Tensor): Edge binary values of shape [num_edges, 1].
            edge_attr (Tensor): Edge feature matrix of shape [num_edges, edge_dim].
            edge_index (Tensor): Edge indices of shape [2, num_edges].

        Returns:
            Tensor: Output values for each node.
        """
        # Ensure x and token have the same number of dimensions
        if self.token_enabled:
            if token.dim() == 1:
                token = token.unsqueeze(-1)  # Convert to [num_nodes, 1]
            elif token.dim() != 2:
                raise ValueError(f"Expected token to have 1 or 2 dimensions, got {token.dim()}")

        if x.dim() == 1:
            x = x.unsqueeze(-1)  # Convert to [num_nodes, 1]
        elif x.dim() != 2:
            raise ValueError(f"Expected x to have 1 or 2 dimensions, got {x.dim()}")

        # Ensure edge_attr is 2D
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)  # Convert to [num_edges, 1]
        elif edge_attr.dim() != 2:
            raise ValueError(f"Expected edge_attr to have 1 or 2 dimensions, got {edge_attr.dim()}")
        
        # Ensure edge_value is 2D
        if edge_value.dim() == 1:
            edge_value = edge_value.unsqueeze(-1)  # Convert to [num_edges, 1]
        elif edge_value.dim() != 2:
            raise ValueError(f"Expected edge_value to have 1 or 2 dimensions, got {edge_value.dim()}")
        
        # Concatenate node features with token features
        if self.token_enabled:
            x = torch.cat([x, token], dim=-1)
                    
        # Combine the original edge attributes with the generated edge values
        combined_edge_attr = torch.cat([edge_attr, edge_value], dim=-1)
        
        # Use the GraphTransformer to compute node outputs with the combined edge attributes
        return self.graph_evaluator_t(x, edge_index, combined_edge_attr), self.graph_evaluator_c(x, edge_index, combined_edge_attr)