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

MAX_TIME_MS = 2000

speed_list_meter_per_s = np.arange(11)/2.
print(speed_list_meter_per_s)
for sidx, speed in enumerate(speed_list_meter_per_s):
    env = WiFiNet(seed=GetSeed(),n_sta=N_TOTAL_STA)
    time_count_ms = 0
    round_counter = 0.
    
    lsh = LSH(num_bits=30, num_tables=20, bits_per_hash=7)
    while True:
        tic = ggm._get_tic()
        agt = agt_for_training()
                
        # get states
        b = env.get_sta_states()
        S_loss, A_loss = env.get_sta_to_associated_ap_loss()

        # tokenize sta states
        l = tk_model.tokenize(b)
        
        hc = sp_model.get_output_np(l)
        hc = sp_model.binarize_hard_code(hc)
        lsh.build_hash_tables(hc)
        adj_tab = lsh.export_adjacency_matrix()

        adj = adj_tab
        adj.eliminate_zeros()
        edge_index = lsh.export_all_edges_of_sparse_matrix(adj)
                
        # pre
        edge_attr = S_loss[edge_index[0], edge_index[1]]
        
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

        tim_us = ggm._get_tim(tic)
        time_count_ms += tim_us/1e3
        env.rand_user_mobility(speed,tim_us,resolution_us=1000)
        WiFiNet.N_PACKETS = math.ceil( tim_us /(nc * agt.TWT_ASLOT_TIME))+1

        ggm._printalltime(f"tim_us:{tim_us}, np:{WiFiNet.N_PACKETS}")

        # run ns3
        ns3sys = sim_sys(id=env.seed)
        ret = ns3sys.step(env=env,agt=agt,sync=True)
        
        bler = WiFiNet.evaluate_bler(ret).squeeze()
        bler = np.mean(bler)
        ggm._add_np_log("rc",sidx,[round_counter,speed])
        ggm._add_np_log("ne",sidx,[edge_index.shape[1],speed])
        ggm._add_np_log("np",sidx,[WiFiNet.N_PACKETS,speed])
        ggm._add_np_log("nc",sidx,[nc,speed])
        ggm._add_np_log("bler",sidx,[bler,speed])
        ggm._add_np_log("tim_us",sidx,[tim_us,speed])

        
        ggm._printalltime(f"rc:{round_counter}, bler:{bler}")
        if time_count_ms >= MAX_TIME_MS:
            break
        round_counter += 1
        
ggm.save_np(LOG_DIR,"final")