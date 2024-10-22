import numpy as np
from scipy.sparse import csr_matrix


from sim_src.ns3_ctrl.sim_sys import sim_sys
from sim_src.ns3_ctrl.wifi_net_ctrl import wifi_net_config
from working_dir_path import get_controller_path, get_ns3_path
from sim_mld.sparse_transformer.tokenizer.model import tokenizer_base
from sim_mld.sparse_transformer.generator.model import VoQ_base
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

VoQ_generator_model = VoQ_base()

N_TRAINING_STEP = 1000
N_BATCH = 20
couter = 0
for i in range(N_TRAINING_STEP):
    batch = []
    sys_list = []
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
        _, edge_index = VoQ_base.get_target_and_edge_index_from_adj_matrix(env.get_CH_matrix())

        # set action
        agt.set_action(np.array([0,0]))
        
        # run ns3
        ns3sys = sim_sys(id=couter)
        ns3sys.step(env=env,agt=agt)
        sys_list.append(ns3sys)
        
        data = {}
        data['token'] = token
        data['edge_index'] = edge_index
        data['ns3sys_id'] = couter
        data['num_nodes'] = n_sta
        batch.append(data)

    
    while sys_list:
        for s in sys_list[:]:  # Iterate over a copy of the list
            if s.is_ns3_end():
                ret = s.wait_ns3_end()
                qos_fail = WiFiNet.evaluate_qos(ret)                
                for d in batch:
                    if d['ns3sys_id'] == s.id:
                        d['target'] = qos_fail
                sys_list.remove(s)
    VoQ_generator_model.step(batch)

VoQ_generator_model.save(GET_LOG_PATH_FOR_SIM_SCRIPT(__file__),"final")
VoQ_generator_model.save_np(GET_LOG_PATH_FOR_SIM_SCRIPT(__file__),"final")  