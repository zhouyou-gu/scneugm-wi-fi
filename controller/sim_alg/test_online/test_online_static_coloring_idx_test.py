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
N_TUNE_STEP = 11

for t in range(N_TRAINING_STEP):
    env = WiFiNet(seed=GetSeed(),n_sta=N_TOTAL_STA)
    agt = agt_for_training()
    lsh = LSH(num_bits=30, num_tables=20, bits_per_hash=7)

    adj_col_list = [csr_matrix((N_TOTAL_STA,N_TOTAL_STA)) for _ in range(20)]
    WiFiNet.N_PACKETS = 1
    QOS_FAIL_MASK = np.zeros(N_TOTAL_STA)
    for i in range(N_TUNE_STEP):
        print(sim_agt_base.TWT_ASLOT_TIME)
        # get states
        b = env.get_sta_states()
        S_loss, A_loss = env.get_sta_to_associated_ap_loss()

        # tokenize sta states
        l = tk_model.tokenize(b)
        # get hard code
        hc = sp_model.get_output_np(l)
        hc = sp_model.binarize_hard_code(hc)
        
        lsh.build_hash_tables(hc)
        adj_tab = lsh.export_adjacency_matrix()
        adj_col = sum(adj_col_list, csr_matrix((N_TOTAL_STA, N_TOTAL_STA)))

        adj = adj_tab + adj_col

        adj.eliminate_zeros()
        edge_index = lsh.export_all_edges_of_sparse_matrix(adj)
        n_processed_edges = edge_index.shape[1]
        print(edge_index.shape[1])
        
        tic = ggm._get_tic()
        # get edge attr
        edge_attr = S_loss[edge_index[0], edge_index[1]]
        # predict o
        oc = pc_model.get_output_np_edge_weight(l,edge_index)
        oh = ph_model.get_output_np_edge_weight(l,edge_index)

        edge_attr = np.stack([edge_attr, oc, oh], axis=-1)  # [num_edges, in_dim_edge]
        
        edge_value = ggm.get_output_np_edge_weight(A_loss, edge_attr, edge_index)
        adj = agt.construct_sparse_adjacency_matrix(edge_index,edge_value,env.n_sta)

        tim = ggm._get_tim(tic)
        print("computational time:", tim)
        adj.eliminate_zeros()
        agt.set_env(env)
        adj_col_list.pop(0)
        adj_col_list.append(adj)
        act = agt.greedy_coloring(adj)
        agt.set_action(act) # place all stas in one slot
        nc = np.max(act)+1
        print(nc,"nc")

        # run ns3
        ns3sys = sim_sys(id=i)
        ret = ns3sys.step(env=env,agt=agt,sync=True)
        qos_fail = WiFiNet.evaluate_qos(ret)
        rwd = 1-qos_fail
        print(np.abs(qos_fail.astype(int)-QOS_FAIL_MASK.astype(int)).sum(),"diff")
        QOS_FAIL_MASK = qos_fail
        print(QOS_FAIL_MASK.sum(),"qos_fail")
        print(act[QOS_FAIL_MASK],"qos_fail_c_idx")
        
        unique_numbers, counts = np.unique(act, return_counts=True)
        result = {int(num): int(count) for num, count in zip(unique_numbers, counts)}
        print(result)
        
        col_idx = np.zeros(100)
        for num, count in zip(unique_numbers, counts):
            col_idx[num] += count
        unique_numbers, counts = np.unique(act[QOS_FAIL_MASK], return_counts=True)
        result = {int(num): int(count) for num, count in zip(unique_numbers, counts)}
        print(result)
        fai_idx = np.zeros(100)
        for num, count in zip(unique_numbers, counts):
            fai_idx[num] += count
        
        ggm._add_np_log("col_idx",i,col_idx,g_step=t)
        ggm._add_np_log("fai_idx",i,fai_idx,g_step=t)
        print(fai_idx)
        print(col_idx)
        
ggm.save_np(LOG_DIR,"final")