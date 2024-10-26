import itertools
import torch
import torch.nn as nn

from sim_src.util import *

from sim_mld.base_model import base_model
from sim_mld.sparse_transformer.transformer.nn import GraphTransformer
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data, Batch

class transformer_base(base_model):
    def __init__(self, MODEL_LR =0.001, THRESHOLD_LR=0.01):
        base_model.__init__(self, LR=MODEL_LR, WITH_TARGET=False)
        self.threshold = to_tensor(np.array(0.))
        self.threshold_lr = THRESHOLD_LR
        self.threshold_avg = 0.5
        # self.threshold_max = 1.
        # self.threshold_min = 0.
        self.threshold_init = False
    def init_model(self):
        self.model = GraphTransformer()

    @counted
    def step(self, batch):
        '''
        batch contains:
            token (K,token_dim): K row vectors representing tokens of each STA 
            edge_index (2,E): the edges between STAs that has the same slot
            target (K): the QoS constraint satisfaction indicator
        '''
        
        if not batch:
            print("None batch in step", self.N_STEP)
            return
        
        state = to_tensor(batch["token"])
        edge_index = to_tensor(batch["edge_index"],dtype=LONG_INTEGER)
        edge_index = to_undirected(edge_index)
        target = to_tensor(batch["target"])

        approx = self.model.forward(state,edge_index).squeeze()

        loss = nn.functional.binary_cross_entropy(approx, target, reduction="mean")

        loss.backward()
        self.model_optim.step()
        self.model_optim.zero_grad()

        worst_target = torch.max(target)
        worst_target = worst_target*2 - 1
        self.threshold -= worst_target*self.threshold_lr
        
        self._printalltime(f"loss: {loss.item():.4f}, threshold: {self.threshold.sigmoid().item():.4f}")
        
        self._add_np_log("loss",self.N_STEP,loss.item())
        self._add_np_log("threshold",self.N_STEP,self.threshold.sigmoid().item())

    @torch.no_grad()
    def get_output_np_graph(self, state, edge_index_np=None):
        if edge_index_np is None:
            K = state.shape[0]
            u, v = np.triu_indices(K, k=1)
            edge_index_np = np.vstack((u, v))
        state = to_tensor(state)
        edge_index_tensor = to_tensor(edge_index_np,dtype=LONG_INTEGER)
        
        
        n_edge = edge_index_tensor.shape[1]
        graphs = []
        # Create individual graph data for each specified pair
        for i, (idx1, idx2) in enumerate(edge_index_tensor.t()):
            # Extract node features for the given index pair
            x_pair = torch.stack([state[idx1], state[idx2]])  # Shape: [2, input_dim]
            edge_index_pair = to_tensor([[0, 1], [1, 0]], dtype=LONG_INTEGER)  # Bidirectional edge between nodes 0 and 1
            graphs.append(Data(x=x_pair, edge_index=edge_index_pair))

        # Batch all pair-wise graphs
        batch = Batch.from_data_list(graphs)
        
        x = self.model.forward(batch.x, batch.edge_index)
        x = x.view(n_edge,2)
        x = to_numpy(x)
        w = np.max(x,axis=1)
        w_max = np.max(w)
        w_min = np.min(w)

            
        w_threhold =  (w_max-w_min)*self.threshold.sigmoid().item() + w_min
        if self.threshold_init:
            self.threshold_avg = self.threshold_avg*(1-self.threshold_lr) + w_threhold*self.threshold_lr
        else:
            self.threshold_avg = w_threhold
            self.threshold_init = True
        
        # w_threhold =  self.threshold.sigmoid().item()
        out = (w >= w_threhold)
        self._printalltime(f"w_max: {w_max:.4f}, w_min: {w_min:.4f}, w_threhold: {w_threhold:.4f}, threshold: {self.threshold.sigmoid().item():.4f}, threshold_avg: {self.threshold_avg:.4f}")
        return edge_index_np[:,out]
        
        # get x
if __name__ == "__main__":
    print(to_tensor(np.array([0.5,1])).sigmoid()>0)