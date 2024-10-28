import torch
import torch.nn as nn
from scipy.sparse import csr_matrix

from sim_src.util import *

from sim_mld.base_model import base_model
from sim_mld.ggm.nn import GraphGenerator
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data, Batch

class E2E_GGM(base_model):
    def __init__(self, LR =0.001):
        base_model.__init__(self, LR=LR)
    
    def init_model(self):
        self.model = GraphGenerator()

    @counted
    def step(self, batch):
        '''
        batch contains:
            state: a list of state vectors
            edge_index (2,E): the edges between STAs that has the same slot
            target_VoQ (E): the VoQ edge value
            target_CnH (E): the CnH edge value
        '''
        
        if not batch:
            print("None batch in step", self.N_STEP)
            return
        # batch states
        train_sequences = [to_tensor(d) for data in batch for d in data['state']]
        padded_train_sequences, train_lengths = pad_tensor_sequence(train_sequences)
        
        # batch VoQ
        data_list = []
        for data in batch:
            target_VoQ = to_tensor(data['target_VoQ'])
            edge_index_VoQ = to_tensor(data['edge_index_VoQ'],dtype=LONG_INTEGER)
            d = Data(x=target_VoQ, edge_index=edge_index_VoQ)
            data_list.append(d)
        VoQ_batch = Batch.from_data_list(data_list)
        
        # batch CnH
        data_list = []
        for data in batch:
            target_CnH = to_tensor(data['target_CnH'])
            edge_index_CnH = to_tensor(data['edge_index_CnH'],dtype=LONG_INTEGER)
            d = Data(x=target_CnH, edge_index=edge_index_CnH)
            data_list.append(d)
        CnH_batch = Batch.from_data_list(data_list)
        
        token = self.model.tokenize(padded_train_sequences,train_lengths).squeeze()

        w_VoQ = self.model.generate_VoQ_graphs(token, VoQ_batch.edge_index).squeeze()
        w_CnH = self.model.generate_CnH_graphs(token, CnH_batch.edge_index).squeeze()
        
        loss_w_VoQ = nn.functional.mse_loss(w_VoQ, VoQ_batch.x, reduction="mean")
        loss_w_CnH = nn.functional.mse_loss(w_CnH, CnH_batch.x, reduction="mean")
   
        
        print(w_VoQ.mean().item(),w_CnH.mean().item(),w_VoQ[VoQ_batch.x.squeeze()==1.].mean().item(),w_CnH[CnH_batch.x.squeeze()==1.].mean().item())
        print(VoQ_batch.x.squeeze().mean().item(),CnH_batch.x.squeeze().mean().item())
        loss = loss_w_VoQ
        loss.backward()
        self.model_optim.step()
        self.model_optim.zero_grad()
        
        self._printalltime(f"loss: {loss.item():.4f}, loss_w_VoQ: {loss_w_VoQ.item():.4f}, loss_w_CnH: {loss_w_CnH.item():.4f}")
        self._add_np_log("loss",self.N_STEP,loss.item())

    @torch.no_grad()
    def get_output_np_edge_weight(self, token, edge_index_np=None):
        if edge_index_np is None:
            K = token.shape[0]
            u, v = np.triu_indices(K, k=1)
            edge_index_np = np.vstack((u, v))
        token = to_tensor(token)
        edge_index_tensor = to_tensor(edge_index_np,dtype=LONG_INTEGER)
        
        w = self.model.forward(token, edge_index_tensor)
        
        return to_numpy(w), edge_index_np
    
    @staticmethod
    def get_target_and_edge_index_from_adj_matrix(A):
        
        """
        Extracts all possible edge indices and their corresponding values from a given adjacency matrix,
        including edges with zero values.

        Parameters:
        -----------
        A : np.ndarray
            The adjacency matrix of the graph (n x n), where n is the number of nodes.

        Returns:
        --------
        edge_indices : np.ndarray
            A (2 x E) matrix containing the source and target node indices for all possible edges,
            where E = n^2.

        edge_values : np.ndarray
            An (E x 1) matrix containing the values on the edges, including zeros.
        """
        n = A.shape[0]
        row_indices, col_indices = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        row_indices = row_indices.flatten()
        col_indices = col_indices.flatten()
        # Exclude self-loops
        mask = row_indices != col_indices
        edge_indices = np.vstack((row_indices[mask], col_indices[mask]))
        edge_values = A[row_indices[mask], col_indices[mask]].reshape(-1, 1)

        return edge_values.squeeze(), edge_indices.squeeze()
   