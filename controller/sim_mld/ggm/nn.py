import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv.transformer_conv import TransformerConv

class EdgeMLP(nn.Module):
    def __init__(self, in_dim_node=6, in_dim_edge=1, hidden_dim=50, num_hidden_layers=3, out_dim=1, activation=nn.GELU(), output_activation=None):
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

class GraphTransformer(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=5, heads=5, num_layers=3, edge_dim=2):
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
        
        # Initial linear layer to project node features to hidden_dim * heads
        self.i_lin = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * heads),
        )
        
        # Create a list of TransformerConv layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                TransformerConv(
                    in_channels=hidden_dim * heads,
                    out_channels=hidden_dim,
                    heads=heads,
                    edge_dim=edge_dim
                )
            )
        
        # Output linear layer to produce final node outputs
        self.o_lin = nn.Sequential(
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

        # Pass the node features through each TransformerConv layer
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr=edge_attr)
        
        # Apply the output linear layer and sigmoid activation to obtain final node outputs
        x = self.o_lin(x)

        return x

class GraphGenerator(nn.Module):
    def __init__(self, node_dim=1, token_dim=5, edge_dim=1):
        """
        Initializes the GraphGenerator with specified dimensions for nodes, tokens, and edges.

        Args:
            node_dim (int): Dimension of the node features.
            token_dim (int): Dimension of the token features.
            edge_dim (int): Dimension of the edge features.
        """
        super(GraphGenerator, self).__init__()
        self.graph_generator = EdgeMLP(
            in_dim_node=node_dim + token_dim,  # Combined node and token dimensions
            in_dim_edge=edge_dim,
            out_dim=2
        )
        self.graph_evaluator = GraphTransformer(
            input_dim=node_dim + token_dim,
            edge_dim=edge_dim + 1  # Including the generated edge value
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
        x = torch.cat([x, token], dim=-1)
        
        # Combine the original edge attributes with the generated edge values
        combined_edge_attr = torch.cat([edge_attr, edge_value], dim=-1)
        
        # Use the GraphTransformer to compute node outputs with the combined edge attributes
        node_outputs = self.graph_evaluator(x, edge_index, combined_edge_attr)
        
        return node_outputs
