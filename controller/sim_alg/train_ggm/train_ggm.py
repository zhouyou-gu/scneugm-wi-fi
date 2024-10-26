import numpy as np
from scipy.sparse import csr_matrix


from sim_src.ns3_ctrl.sim_sys import sim_sys
from sim_src.ns3_ctrl.wifi_net_ctrl import wifi_net_config
from working_dir_path import get_controller_path, get_ns3_path
from sim_mld.ggm.model import E2E_GGM
from sim_src.sim_env.env import WiFiNet
from sim_src.sim_agt.sim_agt_base import sim_agt_base, agt_for_training
from torch_geometric.data import Data, Batch


from sim_src.util import *

np.set_printoptions(precision=3)

wifi_net_config.PROG_PATH = get_ns3_path()
wifi_net_config.PROG_NAME = "sparse-wi-fi/ns3gym_test/env"

ggm = E2E_GGM()

N_TRAINING_STEP = 1000
N_BATCH = 20
n_sta = 2
WiFiNet.N_PACKETS = 1
for i in range(N_TRAINING_STEP):
    batch = []
    n_sta = 2
    for j in range(N_BATCH):
        data = {}
        env = WiFiNet(seed=GetSeed(),n_sta=n_sta)
        agt = agt_for_training()
        agt.set_env(env)
        # get network state 
        state = env.get_sta_states()
        data['state'] = state
        w_CnH, edge_index = E2E_GGM.get_target_and_edge_index_from_adj_matrix(env.get_contending_node_matrix())
        data['target_CnH'] = w_CnH
        data['edge_index_CnH'] = edge_index
        
        # set action
        agt.set_action(np.array([0,0]))
        
        # run ns3
        ns3sys = sim_sys()
        ret = ns3sys.step(env=env,agt=agt,seed=i,sync=True)
        qos_fail = WiFiNet.evaluate_qos(ret)                
        data['target_VoQ'] = qos_fail
        data['edge_index_VoQ'] = edge_index
        batch.append(data)
    
    ggm.step(batch)