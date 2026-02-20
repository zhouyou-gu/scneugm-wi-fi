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

N_CHANNELS = 1

for t in range(N_TRAINING_STEP):
    env = WiFiNet(seed=GetSeed(),n_sta=N_TOTAL_STA)
    
    cell_color = np.zeros((env.cell_size,env.cell_size),dtype=int)
    for i in range(env.cell_size):
        for j in range(env.cell_size):
            cell_color[i][j] = (i+j)%N_CHANNELS
    
    # filter out all STAs in cell_color with channel 0
    cell_color_flat = cell_color.ravel()
    filter = np.zeros(env.n_sta,dtype=int)
    for i in range(int(env.n_sta)):
        sta_loc = env.sta_locs[i]
        cell_x = int(sta_loc[0] // env.cell_edge)
        cell_y = int(sta_loc[1] // env.cell_edge)
        cell_index = cell_x * env.cell_size + cell_y
        if cell_color_flat[cell_index] == 0:
            filter[i] = 1
    
    env.apply_sta_filter(filter)
    
    N_STA = int(env.n_sta)
    print("n_sta:", N_STA)
    agt = agt_for_training()
    lsh = LSH(num_bits=30, num_tables=20, bits_per_hash=7)

    adj_col_list = [csr_matrix((N_STA,N_STA)) for _ in range(20)]
    WiFiNet.N_PACKETS = 1
    QOS_FAIL_MASK = np.zeros(N_STA)
    for i in range(N_TUNE_STEP):
        print(sim_agt_base.TWT_ASLOT_TIME)
        tic = ggm._get_tic()
        # get states
        tim = ggm._get_tim(tic)
        print("computational time:", tim)
        agt.set_env(env)
        act = agt.greedy_coloring(env.get_CH_matrix())
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
        unique_numbers, counts = np.unique(act[QOS_FAIL_MASK], return_counts=True)
        result = {int(num): int(count) for num, count in zip(unique_numbers, counts)}
        print(result)
        unique_numbers, counts = np.unique(act, return_counts=True)
        result = {int(num): int(count) for num, count in zip(unique_numbers, counts)}
        print(result)
        
        ggm._add_np_log("npe",i,[0],g_step=t)
        ggm._add_np_log("nf",i,[QOS_FAIL_MASK.sum()],g_step=t)
        ggm._add_np_log("nc",i,[nc],g_step=t)
        
ggm.save_np(LOG_DIR,"final")