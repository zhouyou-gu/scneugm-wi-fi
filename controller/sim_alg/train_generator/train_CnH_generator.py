import numpy as np
from scipy.sparse import csr_matrix


from sim_src.ns3_ctrl.sim_sys import sim_sys
from sim_src.ns3_ctrl.wifi_net_ctrl import wifi_net_config
from working_dir_path import get_controller_path, get_ns3_path
from sim_mld.sparse_transformer.tokenizer.model import tokenizer_base
from sim_mld.sparse_transformer.generator.model import CnH_base
from sim_src.sim_env.env import WiFiNet
from sim_src.sim_agt.sim_agt_base import sim_agt_base, agt_for_training
from torch_geometric.data import Data, Batch


from sim_src.util import *

np.set_printoptions(precision=3)

wifi_net_config.PROG_PATH = get_ns3_path()
wifi_net_config.PROG_NAME = "sparse-wi-fi/ns3gym_test/env"

# load tokenizer model
tk_model = tokenizer_base()
path = get_controller_path()
path = os.path.join(path, "sim_alg/train_tokenizer/selected_nn/tokenizer_base.final.pt")
tk_model.load_model(path=path)
tk_model.eval()

CnH_generator_model = CnH_base()

N_TRAINING_STEP = 2000
N_BATCH = 20
couter = 0
for i in range(N_TRAINING_STEP):
    batch = []
    n_sta = 2
    for j in range(N_BATCH):
        couter += 1
        env = WiFiNet(seed=couter,n_sta=n_sta)
        agt = agt_for_training()
        agt.set_env(env)
        # get network state 
        b = env.get_sta_states()
        # tokenize sta states
        token, _ = tk_model.get_output_np_batch(b)
        # get collision graph
        target, edge_index = CnH_generator_model.get_target_and_edge_index_from_adj_matrix(env.get_CH_matrix())
        data = {}
        data['token'] = token
        data['edge_index'] = edge_index
        data['target'] = target
        batch.append(data)
        
    CnH_generator_model.step(batch=batch)
    

CnH_generator_model.save(GET_LOG_PATH_FOR_SIM_SCRIPT(__file__),"final")
CnH_generator_model.save_np(GET_LOG_PATH_FOR_SIM_SCRIPT(__file__),"final")  
    
    
    