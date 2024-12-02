import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import vector_to_parameters, parameters_to_vector

COLORING_RELATED_EDGE_DIM =2
HIDDEN_DIM_MULTIPLIER = 5

# do not use softmax to output the binary in es
class EdgeMLP(nn.Module):
    def __init__(self, in_dim_node=1, in_dim_edge=3, hidden_dim=50, num_hidden_layers=2, out_dim=1, activation=nn.ReLU(), output_activation=nn.Sigmoid()):
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


def _flt_param(model):
    return parameters_to_vector(model.parameters())

def _cnt_param(model):
    return sum(param.numel() for param in model.parameters())

def _add_param(model, new_flat_params):
    if new_flat_params.numel() != parameters_to_vector(model.parameters()).numel():
        raise ValueError("Size mismatch: new_flat_params must have the same number of elements as model parameters.")
    
    # Iterator over model parameters
    param_iter = model.parameters()
    pos = 0
    
    for param in param_iter:
        numel = param.numel()
        # Extract the relevant slice from new_flat_params
        new_param_slice = new_flat_params[pos: pos + numel].view_as(param)
        
        # Add the new parameters to the existing ones
        param.data += new_param_slice
        
        pos += numel

def _cpy_param(model, new_flat_params):
    if new_flat_params.numel() != parameters_to_vector(model.parameters()).numel():
        raise ValueError("Size mismatch: new_flat_params must have the same number of elements as model parameters.")
    
    # Iterator over model parameters
    param_iter = model.parameters()
    pos = 0
    
    for param in param_iter:
        numel = param.numel()
        # Extract the relevant slice from new_flat_params
        new_param_slice = new_flat_params[pos: pos + numel].view_as(param)
        
        # Copy the new parameters into the model
        param.data.copy_(new_param_slice)
        
        pos += numel

class ESGraphGenerator(nn.Module):
    def __init__(self, node_dim=1, edge_dim=3, init_v = 0.1):
        """
        Initializes the GraphGenerator with specified dimensions for nodes, tokens, and edges.

        Args:
            node_dim (int): Dimension of the node features.
            token_dim (int): Dimension of the token features.
            edge_dim (int): Dimension of the edge features.
        """
        super(ESGraphGenerator, self).__init__()
        self.init_v = init_v
            
        self.graph_generator_m = EdgeMLP(
            in_dim_node=node_dim,
            in_dim_edge=edge_dim,
        )
        self.graph_generator_v = EdgeMLP(
            in_dim_node=node_dim,
            in_dim_edge=edge_dim,
        )
        self.graph_generator_tmp = EdgeMLP(
            in_dim_node=node_dim,
            in_dim_edge=edge_dim,
        )
        self.n_param = _cnt_param(self.graph_generator_m)
        self._reset_parameters()

    def _reset_parameters(self):
        for param in self.graph_generator_m.parameters():
            nn.init.constant_(param, 0)
        for param in self.graph_generator_v.parameters():
            nn.init.constant_(param, math.log(self.init_v))
        for param in self.graph_generator_tmp.parameters():
            nn.init.constant_(param, 0)
 
    @torch.no_grad()
    def generate_graph(self, x, edge_attr, edge_index):
        """
        Generates a binary value for each edge based on node and edge features.

        Args:
            x (Tensor): Node feature matrix of shape [num_nodes, in_dim_node].
            edge_index (Tensor): Edge indices of shape [2, num_edges].
            edge_attr (Tensor): Edge feature matrix of shape [num_edges, in_dim_edge].

        Returns:
            Tensor: Binary values for each edge of shape [num_edges, 1].
        """
        
        if x.dim() == 1:
            x = x.unsqueeze(-1)  # Convert to [num_nodes, 1]
        elif x.dim() != 2:
            raise ValueError(f"Expected x to have 1 or 2 dimensions, got {x.dim()}")

        # Ensure edge_attr is 2D
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)  # Convert to [num_edges, 1]
        elif edge_attr.dim() != 2:
            raise ValueError(f"Expected edge_attr to have 1 or 2 dimensions, got {edge_attr.dim()}")
        
        # Use the EdgeMLP to generate probabilty for each edge
        edge_values = self.graph_generator_tmp(x, edge_index, edge_attr)  # [num_edges, 2]
        
        return edge_values
    
    @torch.no_grad()
    def update_graph(self, r, LR = 0.01):
        dm = LR*r*(_flt_param(self.graph_generator_tmp)-_flt_param(self.graph_generator_m))/torch.exp(_flt_param(self.graph_generator_v))
        dv = LR*r*(torch.square((_flt_param(self.graph_generator_tmp)-_flt_param(self.graph_generator_m)))/torch.exp(_flt_param(self.graph_generator_v))/2.-1./2.)
        _add_param(self.graph_generator_m, dm)
        _add_param(self.graph_generator_v, dv)

    @torch.no_grad()  
    def update_noise(self):
        epsilon = torch.randn_like(_flt_param(self.graph_generator_m))
        tmp = _flt_param(self.graph_generator_m) + torch.sqrt(torch.exp(_flt_param(self.graph_generator_v))) * epsilon
        _cpy_param(self.graph_generator_tmp, tmp)

    @torch.no_grad()  
    def param_var(self):
        return torch.exp(_flt_param(self.graph_generator_v))
    
    @torch.no_grad()  
    def param_mean(self):
        return _flt_param(self.graph_generator_m)
   
    @torch.no_grad()  
    def no_noise_eval(self):
        _cpy_param(self.graph_generator_tmp, _flt_param(self.graph_generator_m))


if __name__ == "__main__":
    g = ESGraphGenerator()
    e = g.graph_generator_m
    g_np = sum(param.numel() for param in g.parameters())
    e_np = sum(param.numel() for param in e.parameters())
    print("number of params",g_np,e_np)
    print("m",_flt_param(g.graph_generator_m))
    print("v",_flt_param(g.graph_generator_v))
    print("tmp",_flt_param(g.graph_generator_tmp))
    g.update_noise()
    print("m",_flt_param(g.graph_generator_m))
    print("v",_flt_param(g.graph_generator_v))
    print("tmp",_flt_param(g.graph_generator_tmp))
    g.update_graph(1.,1)
    print("m",_flt_param(g.graph_generator_m))
    print("v",_flt_param(g.graph_generator_v))
    print("tmp",_flt_param(g.graph_generator_tmp))
    print(torch.mean(_flt_param(g.graph_generator_tmp)))