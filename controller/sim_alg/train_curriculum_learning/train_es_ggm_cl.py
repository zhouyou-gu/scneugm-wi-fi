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

from sim_mld.curriculum.curr import curr

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

# load GGM
ggm = ES_GGM()
path = get_controller_path()
path = os.path.join(path, "sim_alg/train_and_test_es_with_dhf/selected_nn/ES_GGM.final.pt")
ggm.load_model(path=path)

N_BATCHED_STA = 20
N_TOTAL_STA = 1000
N_TRAINING_STEP = 10000
SAVING_STEPS = [0,125,250,500,1000]

CURR = curr()

WiFiNet.N_PACKETS = 1
for i in range(N_TRAINING_STEP):
    N_STA = CURR.get_n()
    
    env = WiFiNet(seed=GetSeed(),n_sta=N_TOTAL_STA)
    agt = agt_for_training()

    # tokenize sta states
    b = env.get_sta_states()
    l, _ = tk_model.get_output_np_batch(b)
    
    # get hard code
    hc = sp_model.get_output_np(l)
    hc = sp_model.binarize_hard_code(hc)
    
    _, mask = LSH.query_rows(hc,target_matching=N_STA)

    env.apply_sta_filter(mask)

    # tokenize sta states
    b = env.get_sta_states()
    l, _ = tk_model.get_output_np_batch(b)
    
    # pre
    S_loss, A_loss = env.get_sta_to_associated_ap_loss()
    edge_attr, edge_index = agt.export_all_edges(S_loss)
    
    # predict o
    oc = pc_model.get_output_np_edge_weight(l,edge_index)
    oh = ph_model.get_output_np_edge_weight(l,edge_index)

    edge_attr = np.stack([edge_attr, oc, oh], axis=-1)  # [num_edges, in_dim_edge]

    edge_value = ggm.get_output_np_edge_weight(A_loss, edge_attr, edge_index)
    adj = agt.construct_adjacency_matrix(edge_index,edge_value,env.n_sta)
    edge_value_T = adj[edge_index[1],edge_index[0]]
    degree = adj.sum(axis=1)

    agt.set_env(env)
    act = agt.greedy_coloring(adj)
    agt.set_action(act) # place all stas in one slot
    
    ub_act = agt.greedy_coloring(env.get_interfering_node_matrix())
    ub_nc = np.max(ub_act)+1
    color_adj = agt.get_adj_matrix_from_edge_index(env.n_sta,agt.get_same_color_edges(act))
    color_collision, _ = agt.export_all_edges(color_adj)
    nc = np.max(act)+1
    
    # run ns3
    ns3sys = sim_sys(id=i)
    ret = ns3sys.step(env=env,agt=agt,sync=True)
    qos_fail = WiFiNet.evaluate_qos(ret)
    rwd = 1-qos_fail
    
    # rwd = np.zeros(env.n_sta)
    batch = {}
    batch["x"] = A_loss
    batch["token"] = l
    batch["edge_value"] = edge_value
    batch["edge_attr"] = edge_attr
    batch["color_collision"] = color_collision
    batch["edge_index"] = edge_index
    batch["q"] = rwd
    batch["nc"] = nc
    batch["ub_nc"] = ub_nc
    batch["n_sta"] = env.n_sta
    batch["degree"] = degree
    
    ggm.step(batch)
    ggm._add_np_log("n_sta",ggm.N_STEP,[env.n_sta])
    ggm._add_np_log("q_avg",ggm.N_STEP,[rwd.sum()/env.n_sta])
    ggm._add_np_log("nc",ggm.N_STEP,[nc])
    ggm._add_np_log("ub_nc",ggm.N_STEP,[ub_nc])
    
    indicator = (nc<=ub_nc)*((rwd.sum()/env.n_sta)>=ggm.QOS_TARGET)
    CURR.update(float(indicator))
    ggm._printalltime(f"N_STA: {env.n_sta}" )

ggm.save(LOG_DIR,"final")
ggm.save_np(LOG_DIR,"final")