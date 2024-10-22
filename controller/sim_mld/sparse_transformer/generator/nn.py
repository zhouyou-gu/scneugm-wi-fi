import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, TransformedDistribution
from torch.distributions.transforms import SigmoidTransform

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
    
    
    
class JointDistribution(nn.Module):
    """
    A PyTorch module representing a dependent joint distribution over [0,1]^2.
    It uses a Multivariate Normal distribution transformed by Sigmoid to ensure
    that the samples lie within [0,1]^2.
    """
    def __init__(self, initial_mean=None, initial_covariance=None):
        """
        Initializes the JointDistribution module.

        Args:
            initial_mean (torch.Tensor, optional): Initial mean vector of shape [2].
                                                   Defaults to [0, 0].
            initial_covariance (torch.Tensor, optional): Initial covariance matrix of shape [2,2].
                                                         Defaults to identity matrix.
        """
        super(JointDistribution, self).__init__()
        
        # Initialize mean
        if initial_mean is None:
            initial_mean = torch.zeros(2)
        else:
            initial_mean = initial_mean.clone().detach()
        
        # Initialize covariance matrix
        if initial_covariance is None:
            initial_covariance = torch.eye(2)
        else:
            initial_covariance = initial_covariance.clone().detach()
        
        # Make mean a trainable parameter
        self.mean = nn.Parameter(initial_mean)
        
        # Make covariance matrix a trainable parameter
        # To ensure that covariance matrix remains positive definite, we'll parameterize it via its Cholesky factor
        # Here, we initialize it with the Cholesky of the initial covariance
        self.raw_covariance = nn.Parameter(torch.linalg.cholesky(initial_covariance))
    
        self.avg_rwd = nn.Parameter(torch.zeros(1).squeeze())
    
    def get_avg_rwd(self):
        return self.avg_rwd
        
        
    def get_covariance(self):
        """
        Computes the covariance matrix from the raw covariance parameters.

        Returns:
            torch.Tensor: Covariance matrix of shape [2,2].
        """
        # Reconstruct the covariance matrix from the lower-triangular raw_covariance
        # Ensure it's positive definite
        L = self.raw_covariance
        covariance = L @ L.t()
        return covariance
    
    def get_mean(self):
        return self.mean
    
    def forward(self):
        """
        Returns the transformed joint distribution.

        Returns:
            TransformedDistribution: The joint distribution over [0,1]^2.
        """
        # Update the base distribution with current parameters
        base_dist = MultivariateNormal(loc=self.mean, covariance_matrix=self.get_covariance())
        transformed_dist = TransformedDistribution(base_dist, SigmoidTransform())
        return transformed_dist
        
    def sample(self, sample_shape=torch.Size()):
        """
        Samples from the joint distribution.

        Args:
            sample_shape (torch.Size, optional): Shape of the sample. Defaults to torch.Size().

        Returns:
            torch.Tensor: Samples of shape [sample_shape, 2].
        """
        return self.forward().rsample(sample_shape)
    
    def log_prob(self, samples):
        """
        Computes the log probability of given samples.

        Args:
            samples (torch.Tensor): Samples of shape [..., 2] within [0,1]^2.

        Returns:
            torch.Tensor: Log probabilities of shape [...].
        """
        return self.forward().log_prob(samples)