import torch
import torch.nn as nn

from sim_src.util import *

from sim_mld.base_model import base_model
from sim_mld.sparser.nn import hashing_function

class sparser_base(base_model):
    def __init__(self, LR =0.001):
        base_model.__init__(self, LR=LR, WITH_TARGET=False)

    def init_model(self):
        self.model = hashing_function()

    @counted
    def step(self, batch):
        '''
        batch contains:
            token (K,token_dim): K row vectors representing tokens of each STA 
            target_collision_matrix (K,K): the target collision matrix
        '''
        if not batch:
            print("None batch in step", self.N_STEP)
            return
        
        points = to_tensor(batch['token'])

        target_collision_matrix = to_tensor(batch['target_collision_matrix'])
        
        soft_code = self.model(points)

        loss_tot, loss_dis, loss_cor, loss_bal = hashing_function.loss_function(
            soft_code, target_collision_matrix
        )

        loss_tot.backward()
        self.model_optim.step()
        self.model_optim.zero_grad()
        
        self._printalltime(f"loss_tot:{loss_tot:.4f}, loss_dis:{loss_dis:.4f}, loss_cor:{loss_cor:.4f}, loss_bal:{loss_bal:.4f}")
        loss_np = np.array([loss_tot.item(),loss_dis.item(),loss_cor.item(),loss_bal.item()])
        self._add_np_log("loss",self.N_STEP,loss_np)
        
    @torch.no_grad()
    def get_output_np(self, input_np:np.ndarray)->np.ndarray:
        points = to_tensor(input_np)
        hard_code = self.model.hard_code(points)
        return to_numpy(hard_code)
    
    @staticmethod
    def binarize_hard_code(np_hard_code):
        ret = (np_hard_code + 1)/2
        return ret.astype(np.int8)
    
    
