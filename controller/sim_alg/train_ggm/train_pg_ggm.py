import numpy as np
from scipy.sparse import csr_matrix


from sim_src.ns3_ctrl.sim_sys import sim_sys
from sim_src.ns3_ctrl.wifi_net_ctrl import wifi_net_config
from working_dir_path import get_controller_path, get_ns3_path
from sim_mld.pg_ggm.model import PG_GGM
from sim_src.sim_env.env import WiFiNet
from sim_src.sim_agt.sim_agt_base import sim_agt_base, agt_for_training
from torch_geometric.data import Data, Batch

from sim_mld.tokenizer.model import tokenizer_base
from sim_mld.sparser.model import sparser_base
from sim_mld.sparser.lsh import LSH

from sim_src.util import *

np.set_printoptions(precision=3)

wifi_net_config.PROG_PATH = get_ns3_path()
wifi_net_config.PROG_NAME = "sparse-wi-fi/ns3gym_test/env"

# load tokenizer model
tk_model = tokenizer_base()
path = get_controller_path()
path = os.path.join(path, "sim_alg/train_tokenizer/selected_nn/train_tokenizer-2024-October-26-23-49-11-ail/tokenizer_base.final.pt")
tk_model.load_model(path=path)
tk_model.eval()

# load sparser model
sp_model = sparser_base()
path = get_controller_path()
path = os.path.join(path, "sim_alg/train_sparser/log-train_sparser/train_sparser-2024-October-27-03-14-01-ail/sparser_base.final.pt")
sp_model.load_model(path=path)
sp_model.eval()


ggm = PG_GGM()

N_TRAINING_STEP = 10000
WiFiNet.N_PACKETS = 1
def run_test():
    env = WiFiNet(seed=GetSeed(),n_sta=1000)
    
    # tokenize sta states
    b = env.get_sta_states()
    l, _ = tk_model.get_output_np_batch(b)
    
    # get hard code
    hc = sp_model.get_output_np(l)
    hc = sp_model.binarize_hard_code(hc)
    
    _, mask = LSH.query_rows(hc,target_matching=10)


    env.apply_sta_filter(mask)

    # tokenize sta states
    b = env.get_sta_states()
    l, _ = tk_model.get_output_np_batch(b)
    
    S_loss, A_loss = env.get_sta_to_associated_ap_loss()
    edge_attr, edge_index = ggm.export_all_edges(S_loss)
    edge_attr_T, _ = ggm.export_all_edges(S_loss.T)

    edge_value = ggm.get_output_np_edge_weight(A_loss,l,edge_attr, edge_index, edge_attr_T)
    adj = ggm.construct_adjacency_matrix(edge_index,edge_value,env.n_sta)
    edge_value_T = adj[edge_index[1],edge_index[0]]
    degree = adj.sum(axis=1)

    agt = agt_for_training()
    agt.set_env(env)
    act = agt.greedy_coloring(csr_matrix(adj))
    agt.set_action(act) # place all stas in one slot
    
    ub_act = agt.greedy_coloring(csr_matrix(env.get_CH_matrix()))
    ub_nc = np.max(ub_act)+1
    color_adj = agt.get_adj_matrix_from_edge_index(env.n_sta,agt.get_same_color_edges(act)).todense()
    color_adj = np.asarray(color_adj)
    color_collision, _ = ggm.export_all_edges(color_adj)
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
    batch["edge_value_T"] = edge_value_T
    batch["edge_attr"] = edge_attr
    batch["edge_attr_T"] = edge_attr_T
    batch["color_collision"] = color_collision
    batch["edge_index"] = edge_index
    batch["q"] = rwd
    batch["nc"] = nc
    batch["ub_nc"] = ub_nc
    batch["n_sta"] = env.n_sta
    batch["degree"] = degree
    
    ggm.step(batch)


import cProfile
import pstats
import io

# Create a profiler
pr = cProfile.Profile()
pr.enable()  # Start profiling

for i in range(N_TRAINING_STEP):
    run_test()
    
pr.disable()  # Stop profiling

# Create a string stream to capture the profiling data
s = io.StringIO()
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats(10)  # Print the top 10 functions

# Output the profiling results
print(s.getvalue())