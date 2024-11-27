import torch
import torch.nn as nn

from sim_src.util import *

from sim_mld.base_model import base_model
from sim_mld.predictor.nn import predictor

class predictor_base(base_model):
    def __init__(self, LR =0.001):
        base_model.__init__(self, LR=LR, WITH_TARGET=False)

    def init_model(self):
        self.model = predictor()

    @counted
    def step(self, batch):
        if not batch:
            print("None batch in step", self.N_STEP)
            return
        token = to_tensor(batch['token'])
        edge_index = to_tensor(batch["edge_index"],dtype=LONG_INTEGER)
        target = to_tensor(batch["target"])

        predicted = self.model.forward(token,edge_index).squeeze()

        loss = nn.functional.binary_cross_entropy(predicted, target, reduction="mean")
        
        loss.backward()
        self.model_optim.step()
        self.model_optim.zero_grad()

        self._printalltime(f"loss: {loss.item():.4f}")
        self._add_np_log("loss",self.N_STEP,[loss.item()])
  
    @torch.no_grad()      
    def get_output_np_edge_weight(self, token, edge_index):
        token = to_tensor(token)
        edge_index = to_tensor(edge_index,dtype=LONG_INTEGER)
        
        edge_value = self.model.forward(token,edge_index)        
        edge_value = to_numpy(edge_value).squeeze()
        return  edge_value
    
class PCNN(predictor_base):
    pass

class PHNN(predictor_base):
    pass