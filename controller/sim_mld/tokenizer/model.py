import torch
import torch.nn as nn

from sim_src.util import *

from sim_mld.base_model import base_model
from sim_mld.tokenizer.nn import node_tokenizer

class tokenizer_base(base_model):
    PATH_LOSS_MSE_SCALING = 1.
    def __init__(self, LR =0.001):
        base_model.__init__(self, LR=LR, WITH_TARGET=False)

    def init_model(self):
        self.model = node_tokenizer()

    @counted
    def step(self, batch):
        if not batch:
            print("None batch in step", self.N_STEP)
            return
        train_sequences = [to_tensor(d) for d in batch]
        padded_train_sequences, train_lengths = pad_tensor_sequence(train_sequences)
        
        latent_vectors = self.model.encode(padded_train_sequences, train_lengths)
        reconstructed = self.model.decode(latent_vectors, train_lengths)

        
        loss = nn.functional.mse_loss(reconstructed, padded_train_sequences,reduction="none")
        
        mask = torch.zeros_like(loss)
        for i, length in enumerate(train_lengths):
            mask[i, :length, :2] = 1
        loss_lc = (loss * mask).sum() / mask.sum()
        
        mask = torch.zeros_like(loss)
        for i, length in enumerate(train_lengths):
            mask[i, :length, 2] = 1
        loss_pl = (loss * mask).sum() / mask.sum() * self.PATH_LOSS_MSE_SCALING
        
        loss = loss_lc + loss_pl
        loss.backward()
        self.model_optim.step()
        self.model_optim.zero_grad()

        self._printalltime(f"loss: {loss.item():.4f}, loss_lc: {loss_lc.item():.4f}, loss_pl: {loss_pl.item():.4f}")
        self._add_np_log("loss",self.N_STEP,[loss.item(),loss_lc.item(),loss_pl.item()])
  
    @torch.no_grad()      
    def get_output_np(self, input_np:np.ndarray)->np.ndarray:
        padded_train_sequences, train_lengths = pad_tensor_sequence([to_tensor(input_np)])

        latent_vectors = self.model.encode(padded_train_sequences, train_lengths)
        reconstructed = self.model.decode(latent_vectors, train_lengths)

        return to_numpy(latent_vectors[0]), to_numpy(reconstructed[0])

    @torch.no_grad()
    def get_output_np_batch(self, input_np_list)->np.ndarray:
        padded_train_sequences, train_lengths = pad_tensor_sequence([to_tensor(d) for d in input_np_list])

        latent_vectors = self.model.encode(padded_train_sequences, train_lengths)
        reconstructed = self.model.decode(latent_vectors, train_lengths)

        return to_numpy(latent_vectors), to_numpy(reconstructed)
    
    @torch.no_grad()
    def tokenize(self, input_np_list)->np.ndarray:
        padded_train_sequences, train_lengths = pad_tensor_sequence([to_tensor(d) for d in input_np_list])

        latent_vectors = self.model.encode(padded_train_sequences, train_lengths)

        return to_numpy(latent_vectors)