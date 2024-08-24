import os.path

import networkx as nx
import numpy as np
import torch
from torch import optim, nn
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, to_dense_adj

from sim_src.sparse_transformer.model import base_model
from sim_src.sparse_transformer.nn import node_tokenizer, network_performance_transformer
from sim_src.util import USE_CUDA, hard_update_inplace, counted, STATS_OBJECT, soft_update_inplace, to_numpy, to_tensor, \
    to_device, p_true

class tokenizer_base(base_model):
    def init_model(self):
        pass

    @counted
    def step(self, batch):
        pass
