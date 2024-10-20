import torch
import torch.nn as nn

from sim_src.util import *

from sim_mld.base_model import base_model
from sim_mld.sparse_transformer.transformer.nn import GraphTransformer
from torch_geometric.utils import to_undirected

class transformer_base(base_model):
    def __init__(self, LR =0.001):
        base_model.__init__(self, LR=LR, WITH_TARGET=False)
        self.threshold = to_tensor(np.array(0.))
    
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
        edge_index = to_tensor(batch["edge_index"])
        edge_index = to_undirected(edge_index)
        target = to_tensor(batch["target"])

        approx = self.model.forward(state,edge_index)

        loss = nn.functional.mse_loss(target, approx, reduction="mean")

        loss.backward()
        self.model_optim.step()
        self.model_optim.zero_grad()

        self._printalltime(f"loss: {loss.item():.4f}")
        self._add_np_log("loss",self.N_STEP,loss.item())
        
        target = to_tensor(batch["target"])

        worst_target = torch.max(target).detach()
        
        worst_target = (worst_target + 1.)/2. - 1
        
        self.threshold += worst_target*self.LR
        
        
        self._printalltime(f"threshold: {self.threshold.sigmoid().item():.4f}")
        self._add_np_log("threshold",self.N_STEP,self.threshold.sigmoid())


    def get_output_np_graph(self, state, edge_index):
        state = to_tensor(state)
        edge_index = to_tensor(edge_index)
        
        x = self.model.process_pairs(state,edge_index)
        
        x = torch.max(x,dim=1)
        x = x>self.threshold.sigmoid()
        
        out = to_numpy(x).astype(bool)
        
        return edge_index[out]
        
        # get x
if __name__ == "__main__":
    print(to_tensor(np.array([0.5,1])).sigmoid()>0)