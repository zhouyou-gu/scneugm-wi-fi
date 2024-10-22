import numpy as np
from scipy.sparse import csr_matrix


from sim_src.ns3_ctrl.sim_sys import sim_sys
from sim_src.ns3_ctrl.wifi_net_ctrl import wifi_net_config
from working_dir_path import get_controller_path, get_ns3_path
from sim_mld.sparse_transformer.tokenizer.model import tokenizer_base
from sim_mld.sparse_transformer.generator.model import *
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

# load CnH graph model
CnH_generator_model = CnH_base()
path = get_controller_path()
path = os.path.join(path, "sim_alg/train_generator/selected_nn/CnH_base.final.pt")
CnH_generator_model.load_model(path=path)
CnH_generator_model.eval()

# load VoQ graph model
VoQ_generator_model = VoQ_base()
path = get_controller_path()
path = os.path.join(path, "sim_alg/train_generator/selected_nn/VoQ_base.final.pt")
VoQ_generator_model.load_model(path=path)
VoQ_generator_model.eval()

coe_model = graph_interpolator()

N_TRAINING_STEP = 10000
n_sta = 50
WiFiNet.N_PACKETS = 1
for i in range(N_TRAINING_STEP):
    env = WiFiNet(seed=i,n_sta=n_sta)
    agt = agt_for_training()
    agt.set_env(env)
    # get network state 
    b = env.get_sta_states()
    # tokenize sta states
    token, _ = tk_model.get_output_np_batch(b)
    # get collision graph
    w_CnH, edge_index_CnH = CnH_generator_model.get_output_np_edge_weight(token)
    w_VoQ, edge_index_VoQ = VoQ_generator_model.get_output_np_edge_weight(token)
    
    alpha, beta = coe_model.get_interpolation_coefficient()
    adj = coe_model.interpolate_sparse_graphs(w_VoQ,edge_index_VoQ,w_CnH,edge_index_CnH,alpha,beta,num_nodes=n_sta)
    
    act = sim_agt_base.greedy_coloring(adj)
    n_slot = np.max(act)+1
    agt.set_action(act)
    
    ns3sys = sim_sys()
    ret = ns3sys.step(env=env,agt=agt,seed=i,sync=True)

    bler = WiFiNet.evaluate_bler(ret)
    p_qosf = np.sum(bler)/n_sta
    
    batch = {}
    batch['n_slot'] = n_slot
    batch['p_qosf'] = p_qosf
    batch['alpha'] = alpha
    batch['beta'] = beta

    coe_model.step(batch)