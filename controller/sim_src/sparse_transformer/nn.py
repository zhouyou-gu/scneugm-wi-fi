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


class node_tokenizer(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=1):
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

    def decode(self, z, lengths):
        batch_size = z.size(0)
        max_seq_len = lengths.max().item()

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

    def pack_sequence(self, list_of_sequence):
        train_lengths = [seq.size(0) for seq in list_of_sequence]
        padded_train_sequences = pad_sequence(list_of_sequence, batch_first=True)
        lengths = torch.tensor(train_lengths)
        return padded_train_sequences, lengths

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