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
for sidx, speed in enumerate([0]):
    env = WiFiNet(seed=GetSeed(),n_sta=N_TOTAL_STA)
    time_count_ms = 0
    round_counter = 0.
    
    lsh = LSH(num_bits=30, num_tables=20, bits_per_hash=7)
    adj_col_list = [csr_matrix((N_TOTAL_STA,N_TOTAL_STA)) for _ in range(10)]


    tic_tot = ggm._get_tic()
    agt = agt_for_training()
            
    # get states
    b = env.get_sta_states()
    S_loss, A_loss = env.get_sta_to_associated_ap_loss()
    edge_attr, edge_index = agt.export_all_edges(S_loss)

    # tokenize sta states
    tic_tok = ggm._get_tic()
    l = tk_model.tokenize(b)
    tim_tok = ggm._get_tim(tic_tok)
    
    # predict o
    tic_pdt = ggm._get_tic()
    oc = pc_model.get_output_np_edge_weight(l,edge_index)
    oh = ph_model.get_output_np_edge_weight(l,edge_index)
    tim_pdt = ggm._get_tim(tic_pdt)

    # generate edge
    tic_ggm = ggm._get_tic()
    edge_attr = np.stack([edge_attr, oc, oh], axis=-1)  # [num_edges, in_dim_edge]
    edge_value = ggm.get_output_np_edge_weight(A_loss, edge_attr, edge_index)
    adj = agt.construct_sparse_adjacency_matrix(edge_index,edge_value,env.n_sta)
    adj.eliminate_zeros()
    adj_col_list.pop(0)
    adj_col_list.append(adj)
    tim_ggm = ggm._get_tim(tic_ggm)


    agt.set_env(env)

    #coloring
    tic_col = ggm._get_tic()
    act = agt.greedy_coloring(adj)
    agt.set_action(act) # place all stas in one slot
    nc = np.max(act)+1
    tim_col = ggm._get_tim(tic_col)


    tim_tot = ggm._get_tim(tic_tot)
    time_count_ms += tim_tot/1e3
    env.rand_user_mobility(speed,tim_tot,resolution_us=1000)
    WiFiNet.N_PACKETS = math.ceil( tim_tot /(nc * agt.TWT_ASLOT_TIME))+1

    ggm._printalltime(f"tim_tot:{tim_tot}, np:{WiFiNet.N_PACKETS}")

    ggm._add_np_log("tim_tok",sidx,[tim_tok,speed])
    ggm._add_np_log("tim_pdt",sidx,[tim_pdt,speed])
    ggm._add_np_log("tim_ggm",sidx,[tim_ggm,speed])
    ggm._add_np_log("tim_col",sidx,[tim_col,speed])
    ggm._add_np_log("tim_tot",sidx,[tim_tot,speed])

    break
        

ggm.save_np(LOG_DIR,"final")


