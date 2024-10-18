import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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


    
    
if __name__ == "__main__":
    from sim_src.util import to_numpy
    from sim_src.util import to_device
    from sim_src.util import pad_tensor_sequence
    input_dim = 3
    model = node_tokenizer(input_dim=input_dim, hidden_dim=10, latent_dim=10, num_layers=2)
    to_device(model)

    train_sequences = [
        torch.randn(5, input_dim),
        torch.randn(3, input_dim),
        torch.randn(7, input_dim),
        torch.randn(6, input_dim)
    ]
    padded_train_sequences, train_lengths = pad_tensor_sequence(train_sequences)


    criterion = nn.MSELoss(reduction='none')  # We'll handle reduction manually
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 1000
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        # Encode
        latent_vectors = model.encode(padded_train_sequences, train_lengths)
        
        print("seq 1 orignal\n",train_sequences[0][:train_lengths[0]],train_lengths[0])
        
        # Decode
        reconstructed = model.decode(latent_vectors, train_lengths)
        print("seq 1 constructed\n",reconstructed[0][:train_lengths[0]],train_lengths[0])

        # Compute loss
        loss = criterion(reconstructed, padded_train_sequences)
        # Mask the loss to ignore padded positions
        mask = torch.zeros_like(loss)
        for i, length in enumerate(train_lengths):
            mask[i, :length, :] = 1
        loss = (loss * mask).sum() / mask.sum()
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")