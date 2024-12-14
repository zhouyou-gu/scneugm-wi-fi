import numpy as np
from scipy.sparse import csr_matrix


from sim_src.ns3_ctrl.sim_sys import sim_sys
from sim_src.ns3_ctrl.wifi_net_ctrl import wifi_net_config
from working_dir_path import get_controller_path, get_ns3_path
from sim_mld.es_ggm.model import ES_GGM
from sim_src.sim_env.env import WiFiNet
from sim_src.sim_agt.sim_agt_base import sim_agt_base, agt_for_training
from torch_geometric.data import Data, Batch

from sim_mld.tokenizer.model import tokenizer_base
from sim_mld.sparser.model import sparser_base
from sim_mld.sparser.lsh import LSH
from sim_mld.predictor.model import PCNN, PHNN

from sim_src.util import *

LOG_DIR = GET_LOG_PATH_FOR_SIM_SCRIPT(__file__)

np.set_printoptions(precision=3)

wifi_net_config.PROG_PATH = get_ns3_path()
wifi_net_config.PROG_NAME = "sparse-wi-fi/ns3gym_test/env"

# load tokenizer model
tk_model = tokenizer_base()
path = get_controller_path()
path = os.path.join(path, "sim_alg/train_tokenizer/selected_nn/tokenizer_base.final.pt")
tk_model.load_model(path=path)
tk_model.eval()

# load sparser model
sp_model = sparser_base()
path = get_controller_path()
path = os.path.join(path, "sim_alg/train_sparser/selected_nn/sparser_base.final.pt")
sp_model.load_model(path=path)
sp_model.eval()

# load PCNN
pc_model = PCNN()
path = get_controller_path()
path = os.path.join(path, "sim_alg/train_predictor/selected_nn/PCNN.final.pt")
pc_model.load_model(path=path)
pc_model.eval()

# load PHNN
ph_model = PHNN()
path = get_controller_path()
path = os.path.join(path, "sim_alg/train_predictor/selected_nn/PHNN.final.pt")
ph_model.load_model(path=path)
ph_model.eval()

# load sparser model
sp_model = sparser_base()
path = get_controller_path()
path = os.path.join(path, "sim_alg/train_sparser/selected_nn/sparser_base.final.pt")
sp_model.load_model(path=path)
sp_model.eval()

# load GGM
ggm = ES_GGM()
path = get_controller_path()
path = os.path.join(path, "sim_alg/train_curriculum_learning/selected_nn/ES_GGM.final.pt")
ggm.load_model(path=path)
ggm.eval()
# ggm.model.update_noise()


N_TOTAL_STA = 1000
N_TRAINING_STEP = 200

for i in range(N_TRAINING_STEP):
    env = WiFiNet(seed=GetSeed(),n_sta=N_TOTAL_STA)
    agt = agt_for_training()
    lsh = LSH(num_bits=30, num_tables=20, bits_per_hash=7)

    WiFiNet.N_PACKETS = 1
    QOS_FAIL_MASK = np.zeros(N_TOTAL_STA)
    
    # get states
    b = env.get_sta_states()
    S_loss, A_loss = env.get_sta_to_associated_ap_loss()

    # tokenize sta states
    l = tk_model.tokenize(b)

    # pre
    S_loss, A_loss = env.get_sta_to_associated_ap_loss()
    edge_attr, edge_index = agt.export_all_edges(S_loss)
    
    # predict o
    oc = pc_model.get_output_np_edge_weight(l,edge_index)
    oh = ph_model.get_output_np_edge_weight(l,edge_index)

    edge_attr = np.stack([edge_attr, oc, oh], axis=-1)  # [num_edges, in_dim_edge]
    
    edge_value = ggm.get_output_np_edge_weight(A_loss, edge_attr, edge_index)
    adj = agt.construct_sparse_adjacency_matrix(edge_index,edge_value,env.n_sta)

    adj.eliminate_zeros()
    agt.set_env(env)
    act = agt.greedy_coloring(adj)
    agt.set_action(act) # place all stas in one slot
    nc = np.max(act)+1
    print(nc,"nc")

    # run ns3
    ns3sys = sim_sys(id=i)
    ret = ns3sys.step(env=env,agt=agt,sync=True)
    qos_fail = WiFiNet.evaluate_qos(ret)
    rwd = 1-qos_fail
    QOS_FAIL_MASK = qos_fail
    print(QOS_FAIL_MASK.sum(),"qos_fail")
    
    ggm._add_np_log("ae.nf",i,[QOS_FAIL_MASK.sum()])
    ggm._add_np_log("ae.nc",i,[nc])

    # get states
    act = agt.greedy_coloring(env.get_CH_matrix())
    agt.set_action(act) # place all stas in one slot
    nc = np.max(act)+1
    print(nc,"ch_nc")
    # run ns3
    ns3sys = sim_sys(id=i)
    ret = ns3sys.step(env=env,agt=agt,sync=True)
    qos_fail = WiFiNet.evaluate_qos(ret)
    rwd = 1-qos_fail
    QOS_FAIL_MASK = qos_fail
    print(QOS_FAIL_MASK.sum(),"ch_qos_fail")

    ggm._add_np_log("ch.nf",i,[QOS_FAIL_MASK.sum()])
    ggm._add_np_log("ch.nc",i,[nc])
        
       
    # get states
    act = agt.greedy_coloring(env.get_interfering_node_matrix())
    agt.set_action(act) # place all stas in one slot
    nc = np.max(act)+1
    print(nc,"if_nc")
    # run ns3
    ns3sys = sim_sys(id=i)
    ret = ns3sys.step(env=env,agt=agt,sync=True)
    qos_fail = WiFiNet.evaluate_qos(ret)
    rwd = 1-qos_fail
    QOS_FAIL_MASK = qos_fail
    print(QOS_FAIL_MASK.sum(),"if_qos_fail")

    ggm._add_np_log("if.nf",i,[QOS_FAIL_MASK.sum()])
    ggm._add_np_log("if.nc",i,[nc])
        
        
ggm.save_np(LOG_DIR,"final")