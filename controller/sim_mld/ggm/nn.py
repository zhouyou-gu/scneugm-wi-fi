import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv.transformer_conv import *
from torch_geometric.nn import GCNConv

COLORING_RELATED_EDGE_DIM =2

class EdgeMLP(nn.Module):
    def __init__(self, in_dim_node=6, in_dim_edge=1, hidden_dim=30, num_hidden_layers=3, out_dim=1, activation=nn.ReLU(), output_activation=None):
        super(EdgeMLP, self).__init__()
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

    def forward(self, x, edge_index, edge_attr):
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

        return out


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

class GraphTransformer(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=10, heads=5, num_layers=3, edge_dim=1, activation=nn.ReLU(),):
        """
        Initializes the GraphTransformer model.

        Args:
            input_dim (int): Dimension of the input node features.
            hidden_dim (int): Dimension of the hidden layers.
            heads (int): Number of attention heads in the TransformerConv layers.
            num_layers (int): Number of TransformerConv layers.
            edge_dim (int): Dimension of the edge features.
        """
        super(GraphTransformer, self).__init__()
        assert edge_dim>COLORING_RELATED_EDGE_DIM, "edge_dim needs >= 3"

        # Initial linear layer to project node features to hidden_dim * heads
        self.i_lin = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * heads),
            activation,
            nn.Linear(hidden_dim * heads, hidden_dim * heads),
            activation,
            nn.Linear(hidden_dim * heads, hidden_dim * heads)
        )
        self.edge_lin = nn.Sequential(
            nn.Linear(edge_dim-COLORING_RELATED_EDGE_DIM, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Create a list of TransformerConv layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                MaskedTransformerConv(
                    in_channels=hidden_dim * heads,
                    out_channels=hidden_dim,
                    heads=heads,
                    edge_dim=hidden_dim
                )
            )
        
        # Output linear layer to produce final node outputs
        self.o_lin = nn.Sequential(
            nn.Linear(hidden_dim * heads, hidden_dim * heads),
            activation,
            nn.Linear(hidden_dim * heads, hidden_dim * heads),
            activation,
            nn.Linear(hidden_dim * heads, 1),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index, edge_attr):
        """
        Performs a forward pass of the GraphTransformer.

        Args:
            x (Tensor): Node feature matrix of shape [num_nodes, input_dim].
            edge_index (Tensor): Edge indices of shape [2, num_edges].
            edge_attr (Tensor): Edge feature matrix of shape [num_edges, edge_dim].

        Returns:
            Tensor: Output tensor for each node of shape [num_nodes, 1].
        """
        # Project the input node features to the hidden dimension multiplied by the number of heads
        x = self.i_lin(x)
        edge_emb = self.edge_lin(edge_attr[:,:-2])
        edge_attr = torch.cat([edge_emb, edge_attr[:,-2:]], dim=-1)
        # Pass the node features through each TransformerConv layer
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr=edge_attr)
        
        # Apply the output linear layer and sigmoid activation to obtain final node outputs
        x = self.o_lin(x)

        return x

class GCNEvaluator(nn.Module):
    
    def __init__(self, input_dim, edge_dim, hidden_dim=10, num_layers=5, output_dim=1, activation=nn.ReLU()):
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

        self.edge_weight_lin = nn.Linear(edge_dim-COLORING_RELATED_EDGE_DIM, 1)  # <-- New layer to transform edge_attr to edge_weight

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim,add_self_loops=True,normalize=True))

        for _ in range(num_layers - 1):
            self.convs.append(activation)
            self.convs.append(GCNConv(hidden_dim, hidden_dim,add_self_loops=True,normalize=True))

        self.convs.append(nn.Linear(hidden_dim, 1))
        self.convs.append(nn.Sigmoid())
        print(self.convs)
    
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
            mask = edge_attr[:, -2] * (1-edge_attr[:, -1])
            edge_attr_main = edge_attr[:, :-2]
            edge_weight = self.edge_weight_lin(edge_attr_main).squeeze(-1)  # <-- Transform edge_attr to scalar weights
            edge_weight = edge_weight.sigmoid()*mask
        else:
            edge_weight = None

        for layer in self.convs:
            if isinstance(layer, GCNConv):
                x = layer(x, edge_index, edge_weight=edge_weight)  # <-- Pass edge_weight to GCNConv
            else:
                x = layer(x)
        return x

class GraphGenerator(nn.Module):
    def __init__(self, node_dim=1, token_dim=5, edge_dim=1, token_enabled=True):
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
        self.graph_evaluator = GCNEvaluator(
            input_dim=node_dim + token_dim,
            edge_dim=edge_dim + COLORING_RELATED_EDGE_DIM,  # Including the generated edge value and color collision
            hidden_dim=10,
            num_layers=1,
            output_dim=1,
            activation=nn.ReLU()
        )
  
    def generate_graph(self, x, token, edge_attr, edge_index):
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
        
        # Use the EdgeMLP to generate logits for each edge
        edge_logits = self.graph_generator(x, edge_index, edge_attr)  # [num_edges, 2]

        # Apply Gumbel-Softmax to obtain differentiable binary edge values
        edge_values = F.gumbel_softmax(edge_logits, tau=1.0, hard=True)  # [num_edges, 2]

        binary_edge_values = edge_values[:, 1].unsqueeze(-1)  # [200, 1]

        return binary_edge_values

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
        node_outputs = self.graph_evaluator(x, edge_index, combined_edge_attr)
        
        return node_outputs
