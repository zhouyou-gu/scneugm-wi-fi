import torch
import torch.nn as nn

from sim_src.util import *

from sim_mld.base_model import base_model
from sim_mld.es_ggm.nn import *
from torch import optim

class ES_GGM(base_model):
    def __init__(self, LR =0.1):
        base_model.__init__(self, LR=LR, WITH_TARGET = False)
        self.mean_rwd = 0.
    
    def init_model(self):
        self.model = ESGraphGenerator()
    
    def init_optim(self):
        pass
    
    @counted
    def step(self, batch):        
        if not batch:
            print("None batch in step", self.N_STEP)
            return
        
        x = to_tensor(batch["x"])
        token = to_tensor(batch["token"])
        edge_value_action = to_tensor(batch["edge_value"])
        edge_value_T_action = to_tensor(batch["edge_value_T"])
        edge_value_action = torch.logical_or(edge_value_action.bool(), edge_value_T_action.bool()).float()
        edge_attr = to_tensor(batch["edge_attr"])
        edge_attr_T = to_tensor(batch["edge_attr_T"])
        color_collision = to_tensor(batch["color_collision"])
        edge_attr_with_color_collision = torch.cat([edge_attr.unsqueeze(-1),color_collision.unsqueeze(-1)],dim=-1)
        edge_index = to_tensor(batch["edge_index"],dtype=LONG_INTEGER)
        q_target = to_tensor(batch["q"])
        nc = to_tensor(batch["nc"])
        ub_nc = to_tensor(batch["ub_nc"])
        n_sta = to_tensor(batch["n_sta"])
        degree = to_tensor(batch["degree"])

        # c_ratio = torch.clamp(ub_nc/nc,max=1.)
        
        sum_edge_value_action = torch.zeros_like(q_target).scatter_add_(0, edge_index[1], edge_value_action)
        edge_value = self.model.generate_graph(x,token,edge_attr,edge_index,edge_attr_T).squeeze()

        # if q_target.min() == 1:
        #     rwd = ub_nc/nc
        #     rwd = torch.pow(rwd,3/torch.log10(n_sta+1))
        # else:
        #     rwd = q_target.mean()*torch.clamp(ub_nc/nc,max=1.)
        #     rwd = torch.pow(rwd,ub_nc/nc)
        
        if q_target.min() == 1 and nc <= ub_nc:
            rwd = torch.log10(ub_nc/nc)
        else:
            rwd = torch.log10(torch.clamp(q_target.mean()*torch.clamp(ub_nc/nc,max=1.),min=1e-5))
        
        
        self.mean_rwd = 0.1*rwd.item() + self.mean_rwd*0.9
        self.model.update_graph(rwd.item()-self.mean_rwd,self.LR)

        s = ""
        s += f"K:{q_target.sum():>4.0f}/{n_sta.item():>4.0f}:{nc.item():>3.0f}>{ub_nc.item():>3.0f}"
        s += f", e:{edge_value.mean():>6.2f}"
        s += f", E:{edge_value_action.numel():>6d}"
        s += f", e+:{edge_value_action[edge_value_action>0].sum():>6.0f}"
        s += f", i+:{(edge_attr>0).sum():>6.0f}"
        s += f", s_e|i+:{edge_value_action[edge_attr>0].sum():>6.0f}"
        s += f", m_e:{edge_value_action.mean():>6.2f}"
        s += f", m_d|q+:{sum_edge_value_action[q_target>0].mean():>6.2f}"
        s += f", m_d|q0:{sum_edge_value_action[q_target==0].mean():>6.2f}"
        s += f", m_d/K|q+:{sum_edge_value_action[q_target>0].mean()/n_sta.item():>6.2f}"
        s += f", m:{self.model.param_mean().mean():>6.4f}"
        s += f", mv:{self.model.param_mean().var():>6.4f}"
        s += f", vm:{self.model.param_var().mean():>6.4f}"
        s += f", v_:{self.model.param_var().min():>6.4f}"
        s += f", v^:{self.model.param_var().max():>6.4f}"



        # s += f", c(e,i):{torch.corrcoef(torch.stack([edge_value, edge_attr>0]))[0, 1].item():>4.2f}"
        # s += f", loss_gen:{loss_gen.item():>4.2f}"
        self._printalltime(s)

        # self._printalltime(f"loss_eva: {loss_eva.item():.4f}, loss_gen: {loss_gen.item():.4f}")
        # self._add_np_log("loss",self.N_STEP,loss_eva.item(),loss_gen.item())

        self.model.update_noise()


    @torch.no_grad()
    def get_output_np_edge_weight(self, x, token, edge_attr, edge_index, edge_attr_T, hard = True):
        x = to_tensor(x)
        token = to_tensor(token)
        edge_attr = to_tensor(edge_attr)
        edge_index = to_tensor(edge_index,dtype=LONG_INTEGER)
        edge_attr_T = to_tensor(edge_attr_T)
        
        edge_value = self.model.generate_graph(x,token,edge_attr,edge_index, edge_attr_T)        
        edge_value = to_numpy(edge_value).squeeze()
        if not hard:
            edge_value = binarize_vector(edge_value)
        else:
            edge_value = edge_value > 0.5
            edge_value = edge_value.astype(float)
        return  edge_value
     
 