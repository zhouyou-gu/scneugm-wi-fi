import math
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.distributions import MultivariateNormal, TransformedDistribution, SigmoidTransform, Normal

class node_tokenizer(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=10, latent_dim=5, num_layers=1):
        super(node_tokenizer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # Latent space
        self.latent = nn.Linear(hidden_dim, latent_dim)
        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True)
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, lengths):
        # Encode
        z = self.encode(x, lengths)
        # Decode
        output = self.decode(z, lengths)
        return output

    def encode(self, x, lengths):
        # Pack the padded sequence
        packed_x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        
        # Encoder
        _, (h_n, _) = self.encoder_lstm(packed_x)
        # h_n: [num_layers, batch_size, hidden_dim]
        h_n = h_n[-1]  # Take the last layer's hidden state
        # h_n: [batch_size, hidden_dim]
        
        # Latent space
        z = self.latent(h_n)
        # z: [batch_size, latent_dim]
        return z

    def decode(self, z, lengths:list):
        batch_size = z.size(0)
        max_seq_len = max(lengths)

        # Prepare decoder inputs
        decoder_input = torch.zeros(batch_size, max_seq_len, z.size(1), device=z.device)
        for i, length in enumerate(lengths):
            decoder_input[i, :length, :] = z[i].unsqueeze(0).repeat(length, 1)
        
        # Pack the decoder inputs
        packed_decoder_input = pack_padded_sequence(decoder_input, lengths, batch_first=True, enforce_sorted=False)
        
        # Decoder
        packed_decoder_output, _ = self.decoder_lstm(packed_decoder_input)
        
        # Unpack the decoder outputs
        decoder_output, _ = pad_packed_sequence(packed_decoder_output, batch_first=True, total_length=max_seq_len)
        # decoder_output: [batch_size, max_seq_len, hidden_dim]
        
        # Output
        output = self.output_layer(decoder_output)
        # output: [batch_size, max_seq_len, input_dim]
        
        return output




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
    A PyTorch module representing an independent joint distribution over [0,1]^2.
    It uses two separate Normal distributions transformed by Sigmoid to ensure
    that the samples lie within [0,1]^2.
    """
    def __init__(self, initial_mean=None, initial_std=None):
        """
        Initializes the JointDistribution module.

        Args:
            initial_mean (torch.Tensor, optional): Initial mean vector of shape [2].
                                                   Defaults to [0, 0].
            initial_std (torch.Tensor, optional): Initial standard deviations of shape [2].
                                                  Defaults to [1, 1].
        """
        super(JointDistribution, self).__init__()
        
        # Initialize means
        if initial_mean is None:
            initial_mean = torch.zeros(2)
        else:
            initial_mean = initial_mean.clone().detach()
        
        # Initialize standard deviations
        if initial_std is None:
            initial_std = torch.ones(2)
        else:
            initial_std = initial_std.clone().detach()
        
        # Make means and log standard deviations trainable parameters
        self.mean = nn.Parameter(initial_mean)
        self.log_std = nn.Parameter(torch.log(initial_std))  # Parameterize in log-space for positivity
        
        self.avg_rwd = nn.Parameter(torch.zeros(1).squeeze())

    def get_avg_rwd(self):
        return self.avg_rwd
        
    def get_std(self):
        """
        Computes the standard deviations from the log_std parameters.

        Returns:
            torch.Tensor: Standard deviations of shape [2].
        """
        return torch.exp(self.log_std)
    
    def get_variance(self):
        """
        Computes the variances from the log_std parameters.

        Returns:
            torch.Tensor: Variances of shape [2].
        """
        return self.get_std() ** 2

    def get_mean(self):
        return self.mean
    
    def forward(self):
        """
        Returns the transformed joint distribution.

        Returns:
            TransformedDistribution: The joint distribution over [0,1]^2.
        """
        # Define independent Normal distributions for each dimension
        base_dist = MultivariateNormal(loc=self.mean, covariance_matrix=torch.diag(self.get_variance()))
        
        # Apply Sigmoid transform to ensure outputs are in [0,1]^2
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
    
    def log_prob_separately(self, samples):
        """
        Computes the log probabilities of each variable separately.

        Args:
            samples (torch.Tensor): Samples of shape [..., 2] within [0,1]^2.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - Log probabilities of the first variable of shape [...].
                - Log probabilities of the second variable of shape [...].
        """
        if samples.shape[-1] != 2:
            raise ValueError("Samples must have the last dimension of size 2.")
        
        # Split the samples into two variables
        samples1 = samples[..., 0]
        samples2 = samples[..., 1]
        
        # Define TransformedDistribution for the first variable
        base_dist1 = Normal(loc=self.mean[0], scale=self.get_std()[0])
        trans_dist1 = TransformedDistribution(base_dist1, SigmoidTransform())
        log_prob1 = trans_dist1.log_prob(samples1)
        
        # Define TransformedDistribution for the second variable
        base_dist2 = Normal(loc=self.mean[1], scale=self.get_std()[1])
        trans_dist2 = TransformedDistribution(base_dist2, SigmoidTransform())
        log_prob2 = trans_dist2.log_prob(samples2)
        
        return log_prob1, log_prob2