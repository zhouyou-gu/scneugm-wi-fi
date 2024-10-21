import numpy as np
from scipy.sparse import csr_matrix


from sim_src.ns3_ctrl.sim_sys import sim_sys
from sim_src.ns3_ctrl.wifi_net_ctrl import wifi_net_config
from working_dir_path import get_controller_path, get_ns3_path
from sim_mld.sparse_transformer.tokenizer.model import tokenizer_base
from sim_mld.sparse_transformer.transformer.model import transformer_base
from sim_src.sim_env.env import WiFiNet
from sim_src.sim_agt.sim_agt_base import sim_agt_base, agt_for_training


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

tf_model = transformer_base()


N_TRAINING_STEP = 1000
N_CORES = 1 # number of cores to run ns3 in parallel
couter = 0
for i in range(N_TRAINING_STEP):
    sys_list = []
    for j in range(N_CORES):
        couter += 1
        env = WiFiNet(seed=couter)
        agt = agt_for_training()
        agt.set_env(env)
        
        # get network state 
        b = env.get_sta_states()
        # tokenize sta states
        token, _ = tk_model.get_output_np_batch(b)
        
        # get collision graph
        edges = tf_model.get_output_np_graph(token)
        adj = sim_agt_base.get_adj_matrix_from_edge_index(env.n_sta,edges)
        act = sim_agt_base.greedy_coloring(adj)
        agt.set_action(act)
        
        # run ns3
        ns3sys = sim_sys()
        ns3sys.step(env=env,agt=agt)
        sys_list.append(ns3sys)
    
    while sys_list:
        for s in sys_list[:]:  # Iterate over a copy of the list
            if not s.is_ns3_end():
                ret = s.wait_ns3_end()
                qos_fail = WiFiNet.evaluate_qos(ret)

                print(f"failed user percentage: {np.sum(qos_fail)/env.n_sta:.2f}, total slots:{np.max(act):4d}")
                    
                batch = {}
                batch["token"] = token
                batch["edge_index"] = sim_agt_base.get_same_color_edges(act)
                batch["target"] = qos_fail
                tf_model.step(batch)
                sys_list.remove(s)

    
    