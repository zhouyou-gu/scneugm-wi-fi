import math
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.distributions import MultivariateNormal, TransformedDistribution, SigmoidTransform, Normal
import torch.nn.functional as F

class node_tokenizer(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=10, latent_dim=5, num_layers=1):
        super(node_tokenizer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        
        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
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
        x =  self.input_layer(x)
        
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

class GraphGenerator(nn.Module):
    def __init__(self):
        super(GraphGenerator, self).__init__()
        self.tokenizer = node_tokenizer()
        self.VoQ = EdgeMLP()
        self.CnH = EdgeMLP()

    def tokenize(self, x, lengths):
        token = self.tokenizer.encode(x, lengths)
        return token
        
    def generate_VoQ_graphs(self, token, edge_index):
        return self.VoQ.forward(token, edge_index)

    def generate_CnH_graphs(self, token, edge_index):
        return self.CnH.forward(token, edge_index)
    

