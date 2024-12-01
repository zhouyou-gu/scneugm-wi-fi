import torch
import torch.nn as nn
from scipy.sparse import csr_matrix

from sim_src.util import *

from sim_mld.base_model import base_model
from sim_mld.pg_ggm.nn import GraphGenerator
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data, Batch
from torch import optim
torch.autograd.set_detect_anomaly(True)

class PG_GGM(base_model):
    QOS_TARGET = 0.99
    def __init__(self, LR =0.001, deterministic=True):
        base_model.__init__(self, LR=LR, WITH_TARGET = False)
        self.deterministic = deterministic
        self.mean_rwd = 0.

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
        edge_attr = to_tensor(batch["edge_attr"])
        color_collision = to_tensor(batch["color_collision"])
        edge_index = to_tensor(batch["edge_index"],dtype=LONG_INTEGER)
        q_target = to_tensor(batch["q"])
        nc = to_tensor(batch["nc"])
        ub_nc = to_tensor(batch["ub_nc"])
        n_sta = to_tensor(batch["n_sta"])
        degree = to_tensor(batch["degree"])
        
        sum_edge_value_action = torch.zeros_like(q_target).scatter_add_(0, edge_index[1], edge_value_action)

        if q_target.min() == 1:
            rwd = torch.log10(ub_nc/nc)
        else:
            rwd = torch.log10(torch.clamp(torch.clamp(q_target/self.QOS_TARGET,max=1.).mean() *torch.clamp(ub_nc/nc,max=1.),min=1e-5))

        if self.deterministic:
            # c_ratio = torch.log10(ub_nc/nc)
            q_approx_action = self.model.evaluate_graph(x,edge_value_action,edge_attr,edge_index)
            q_approx_action = q_approx_action.squeeze()
            # loss_eva = nn.functional.binary_cross_entropy(q_approx_action_t, q_target, reduction="mean")
            rwd_v = torch.ones_like(q_approx_action)*rwd
            
            loss_eva = nn.functional.mse_loss(q_approx_action, rwd_v, reduction="mean")
            self.eva_optim.zero_grad()
            loss_eva.backward()
            self.eva_optim.step()
            self.eva_optim.zero_grad()
            
            edge_value = self.model.generate_graph(x,edge_attr,edge_index).squeeze()
            q_approx = self.model.evaluate_graph(x,edge_value,edge_attr,edge_index)
            q_approx = q_approx.squeeze()
            loss_gen = -q_approx.mean()
        else:
            edge_value = self.model.generate_graph(x,edge_attr,edge_index).squeeze()
            edge_value = torch.clamp(edge_value,min=1e-5)
            self.mean_rwd = 0.1*rwd.item() + self.mean_rwd*0.9
            loss_gen = - torch.log(edge_value).mean() * (rwd.item()-self.mean_rwd)
            
        I = edge_attr[:,0]
        s = ""
        s += f"K:{q_target.sum():>4.0f}/{n_sta.item():>4.0f}:{nc.item():>3.0f}>{ub_nc.item():>3.0f}"
        s += f", e:{edge_value.mean():>6.2f}"
        s += f", E:{edge_value_action.numel():>6d}"
        s += f", e+:{edge_value_action[edge_value_action>0].sum():>6.0f}"
        s += f", i+:{(I>0).sum():>6.0f}"
        s += f", s_e|i+:{edge_value_action[I>0].sum():>6.0f}"
        s += f", m_e:{edge_value_action.mean():>6.2f}"
        s += f", m_d|q+:{sum_edge_value_action[q_target>0].mean():>6.2f}"
        s += f", m_d|q0:{sum_edge_value_action[q_target==0].mean():>6.2f}"
        s += f", m_d/K|q+:{sum_edge_value_action[q_target>0].mean()/n_sta.item():>6.2f}"

        self._printalltime(s)
        self._add_np_log("reward",self.N_STEP,[rwd.item()])

        self.gen_optim.zero_grad()
        loss_gen.backward()
        self.gen_optim.step()
        self.gen_optim.zero_grad()

    @torch.no_grad()
    def get_output_np_edge_weight(self, x, edge_attr, edge_index, hard = False):
        x = to_tensor(x)
        edge_attr = to_tensor(edge_attr)
        edge_index = to_tensor(edge_index,dtype=LONG_INTEGER)
                
        edge_value = self.model.generate_graph(x,edge_attr,edge_index)        
        edge_value = to_numpy(edge_value).squeeze()
        if not hard:
            edge_value = binarize_vector(edge_value)
        else:
            edge_value = edge_value > 0.5
            edge_value = edge_value.astype(float)
        return  edge_value
 
 
    @torch.no_grad()
    def get_output_np_edge_weight_raw(self, x, edge_attr, edge_index):
        x = to_tensor(x)
        edge_attr = to_tensor(edge_attr)
        edge_index = to_tensor(edge_index,dtype=LONG_INTEGER)
                
        edge_value = self.model.generate_graph(x,edge_attr,edge_index)        
        edge_value = to_numpy(edge_value).squeeze()
        return  edge_value