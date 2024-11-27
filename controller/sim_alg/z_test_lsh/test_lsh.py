import numpy as np
import scipy
from scipy.sparse import csr_matrix

from sim_src.ns3_ctrl.wifi_net_ctrl import wifi_net_config
from sim_src.ns3_ctrl.sim_sys import sim_sys
from sim_src.sim_agt.sim_agt_base import agt_for_training
from working_dir_path import get_controller_path, get_ns3_path
from sim_mld.sparser.model import sparser_base
from sim_mld.sparser.lsh import LSH
from sim_mld.tokenizer.model import tokenizer_base
from sim_src.sim_env.env import WiFiNet

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


N_TRAINING_STEP = 10
WiFiNet.N_PACKETS = 1
for i in range(N_TRAINING_STEP):
    # get network state 
    env = WiFiNet(seed=GetSeed(),n_sta=1000)
    b = env.get_sta_states()
    
    # tokenize sta states
    l, _ = tk_model.get_output_np_batch(b)
    
    # get collision matrix
    target_collision_matrix = env.get_CH_matrix()
    target_collision_matrix = csr_matrix(target_collision_matrix).astype(np.int8)
    target_collision_matrix.eliminate_zeros()
    
    # get hard code
    hc = sp_model.get_output_np(l)
    hc = sp_model.binarize_hard_code(hc)
    print(hc.shape)
    # lsh
    lsh = LSH(num_bits=sp_model.model.hash_dim, num_tables=30, bits_per_hash=10)
    lsh.build_hash_tables(hc)
    hamming_distance_threhold = None
    approx_collision_matrix = lsh.export_adjacency_matrix(hamming_distance_threhold)
    print(approx_collision_matrix.nnz,hamming_distance_threhold)
    # approx_collision_matrix = target_collision_matrix.multiply(approx_collision_matrix)
    # approx_collision_matrix.eliminate_zeros()

    res = lsh.compare_adjacency_matrices(approx_collision_matrix,target_collision_matrix)
    print(res)

    agt = agt_for_training()
    agt.set_env(env)
    act = agt.greedy_coloring(approx_collision_matrix)
    agt.set_action(act)

    ns3sys = sim_sys()
    ret = ns3sys.step(env=env,agt=agt,seed=env.seed,sync=True)
    qos_fail = WiFiNet.evaluate_qos(ret)   
    print(np.mean(qos_fail),np.max(act)+1)
        
    act = agt.greedy_coloring(env.get_CH_matrix())
    agt.set_action(act)
    ns3sys = sim_sys()
    ret = ns3sys.step(env=env,agt=agt,seed=env.seed,sync=True)
    qos_fail = WiFiNet.evaluate_qos(ret)   
    print(np.mean(qos_fail),np.max(act)+1)