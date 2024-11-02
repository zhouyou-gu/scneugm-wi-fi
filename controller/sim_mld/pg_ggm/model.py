import torch
import torch.nn as nn
from scipy.sparse import csr_matrix

from sim_src.util import *

from sim_mld.base_model import base_model
from sim_mld.pg_ggm.nn import GraphGenerator
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data, Batch
from torch import optim
import networkx as nx

class PG_GGM(base_model):
    def __init__(self, LR =0.001, deterministic=True):
        base_model.__init__(self, LR=LR, WITH_TARGET = False)
        self.deterministic = deterministic
    
    def init_model(self):
        self.model = GraphGenerator()
    
    def init_optim(self):
        self.eva_optim = optim.Adam(list(self.model.graph_evaluator_t.parameters()) + list(self.model.graph_evaluator_c.parameters()), lr=self.LR)
        self.gen_optim = optim.Adam(self.model.graph_generator.parameters(), lr=self.LR)

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

        c_ratio = torch.clamp(ub_nc/degree,max=1.)
       
        q_approx_action_t, q_approx_action_c = self.model.evaluate_graph(x,token,edge_value_action,edge_attr_with_color_collision,edge_index)
        q_approx_action_t = q_approx_action_t.squeeze()
        q_approx_action_c = q_approx_action_c.squeeze()
        sum_edge_value_action = torch.zeros_like(q_approx_action_t).scatter_add_(0, edge_index[1], edge_value_action)
        c_ratio_target = torch.zeros_like(q_approx_action_c) + c_ratio
        c_ratio_target = (c_ratio_target>=1.).float()
        loss_eva = nn.functional.binary_cross_entropy(q_approx_action_t, q_target, reduction="mean")
        loss_eva += nn.functional.binary_cross_entropy(q_approx_action_c, c_ratio_target, reduction="mean")
        self.eva_optim.zero_grad()
        loss_eva.backward()
        self.eva_optim.step()
        self.eva_optim.zero_grad()


        edge_value = self.model.generate_graph(x,token,edge_attr,edge_index,edge_attr_T).squeeze()
        q_approx, q_approx_c = self.model.evaluate_graph(x,token,edge_value,edge_attr_with_color_collision,edge_index)
        q_approx = q_approx.squeeze()
        q_approx_c = q_approx_c.squeeze()
        # sum_edge_value = torch.zeros_like(q_approx).scatter_add_(0, edge_index[1], edge_value)
        # loss_gen = nn.functional.mse_loss(edge_value,(edge_attr>0).float(), reduction="mean")

        if self.deterministic:
            loss_gen = -(q_approx * q_approx_c).mean() 
        else:
            edge_value = torch.clamp(edge_value,min=1e-5)
            loss_gen = - torch.log(edge_value).mean() * ((q_target*c_ratio_target).mean()-(q_approx * q_approx_c).mean())
        
        self.gen_optim.zero_grad()
        loss_gen.backward()
        self.gen_optim.step()
        self.gen_optim.zero_grad()
        
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
        s += f", m_d/K|q0:{sum_edge_value_action[q_target==0].mean()/n_sta.item():>6.2f}"
        s += f", q~:{q_approx.mean():>6.2f}"
        s += f", ql:{loss_eva.item():>6.2f}"
        # s += f", c(e,i):{torch.corrcoef(torch.stack([edge_value, edge_attr>0]))[0, 1].item():>4.2f}"
        # s += f", loss_gen:{loss_gen.item():>4.2f}"
        self._printalltime(s)

        # self._printalltime(f"loss_eva: {loss_eva.item():.4f}, loss_gen: {loss_gen.item():.4f}")
        self._add_np_log("loss",self.N_STEP,loss_eva.item(),loss_gen.item())

    @torch.no_grad()
    def get_output_np_edge_weight(self, x, token, edge_attr, edge_index, edge_attr_T, hard = False):
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
     
 