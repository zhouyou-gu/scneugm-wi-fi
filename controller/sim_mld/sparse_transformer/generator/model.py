import torch
import torch.nn as nn
from scipy.sparse import csr_matrix

from sim_src.util import *

from sim_mld.base_model import base_model
from sim_mld.sparse_transformer.generator.nn import EdgeMLP, JointDistribution
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data, Batch

class CnH_base(base_model):
    def __init__(self, LR =0.001):
        base_model.__init__(self, LR=LR)
    
    def init_model(self):
        self.model = EdgeMLP()

    @counted
    def step(self, batch:Batch):
        '''
        batch contains:
            token (K,token_dim): K row vectors representing tokens of each STA 
            edge_index (2,E): the edges between STAs that has the same slot
            target (E): the CnH edge value
        '''
        
        if not batch:
            print("None batch in step", self.N_STEP)
            return
        
        data_list = []
        for data in batch:
            token = to_tensor(data['token'])
            edge_index = to_tensor(data['edge_index'],dtype=LONG_INTEGER)
            target = to_tensor(data['target'])
            d = Data(token=token,edge_index=edge_index,target=target)
            data_list.append(d)
        
        batch = Batch.from_data_list(data_list)
        approx = self.model.forward(batch.token,batch.edge_index).squeeze()

        loss = nn.functional.binary_cross_entropy(approx, batch.target, reduction="mean")

        loss.backward()
        self.model_optim.step()
        self.model_optim.zero_grad()
        
        self._printalltime(f"loss: {loss.item():.4f}")
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
    
class VoQ_base(CnH_base):
    pass

class graph_interpolator(base_model):
    def __init__(self, LR =0.01):
        base_model.__init__(self, LR=LR)
    
    def init_model(self):
        self.model = JointDistribution()
        
        
    @counted
    def step(self, batch):
        '''
        batch contains:
            n_slot: the number of slots by coloring the graph
            p_qosf: the percentage of users fails in QoS
            alpha: weights between two graphs
            beta: the threhold
        '''
        
        if not batch:
            print("None batch in step", self.N_STEP)
            return
        n_slot = batch['n_slot']
        p_qosf = batch['p_qosf']
        alpha = batch['alpha']
        beta = batch['beta']
        param = to_tensor(np.array([alpha,beta]))
        
        log_prob = self.model.log_prob(param).squeeze()

        rwd = self.get_rwd(n_slot,p_qosf)-self.model.get_avg_rwd().detach()
        loss = - rwd*log_prob
        loss.backward()
        self.model_optim.step()
        self.model_optim.zero_grad()
        
        self._printalltime(f"loss: {loss.item():.4f}, rwd: {rwd:.4f}, n_slot: {n_slot:.4f}, p_qosf: {p_qosf:.4f}, alpha: {alpha:.4f}, beta: {beta:.4f}, ")
        self._printalltime(f"mean: {to_numpy(self.model.get_mean())}, variance: {to_numpy(self.model.get_covariance()).flatten()}")
        self._add_np_log("loss",self.N_STEP,loss.item())
        
        l = nn.functional.mse_loss(self.model.get_avg_rwd(),rwd)
        l.backward()
        self.model_optim.step()
        self.model_optim.zero_grad()

       

    @torch.no_grad()
    def get_rwd(self, n_slot, p_qosf):
        if p_qosf > 0:
            return to_tensor(n_slot)
        else:
            return - to_tensor(n_slot)

    @torch.no_grad()
    def get_interpolation_coefficient(self):
        coe = self.model.sample()
        coe = to_numpy(coe)
        return coe[0], coe[1]
    
    
    @staticmethod
    def interpolate_sparse_graphs(w_VoQ,edge_index_VoQ,w_CnH,edge_index_CnH, alpha, beta, num_nodes=None):
        w_VoQ = w_VoQ.flatten()
        w_CnH = w_CnH.flatten()

        if num_nodes is None:
            num_nodes = max(
                edge_index_VoQ.max() if edge_index_VoQ.size > 0 else -1,
                edge_index_CnH.max() if edge_index_CnH.size > 0 else -1
            ) + 1

        A1 = csr_matrix(
            (w_VoQ, (edge_index_VoQ[0], edge_index_VoQ[1])),
            shape=(num_nodes, num_nodes)
        )

        A2 = csr_matrix(
            (w_CnH, (edge_index_CnH[0], edge_index_CnH[1])),
            shape=(num_nodes, num_nodes)
        )

        A = (1 - alpha) * A1 + alpha * A2

        # Copy the sparse matrix structure
        B = A.copy()
        
        # Threshold the data
        B.data = (B.data >= beta).astype(int)
        
        # Eliminate zeros to maintain sparse structure
        B.eliminate_zeros()
        return B
    