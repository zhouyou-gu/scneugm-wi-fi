import numpy as np
from scipy.sparse import csr_matrix


from working_dir_path import get_controller_path
from sim_mld.sparse_transformer.sparser.model import sparser_base
from sim_mld.sparse_transformer.sparser.lsh import LSH
from sim_mld.sparse_transformer.tokenizer.model import tokenizer_base
from sim_src.sim_env.env import WiFiNet

from sim_src.util import *

np.set_printoptions(precision=3)

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


N_TRAINING_STEP = 10
for i in range(N_TRAINING_STEP):
    # get network state 
    e = WiFiNet(seed=i)
    b = e.get_sta_states()
    
    # tokenize sta states
    l, _ = tk_model.get_output_np_batch(b)
    
    # get collision matrix
    target_collision_matrix = e.get_CH_matrix()
    target_collision_matrix = csr_matrix(target_collision_matrix).astype(np.int8)
    target_collision_matrix.eliminate_zeros()
    
    # get hard code
    hc = sp_model.get_output_np(l)
    hc = sp_model.binarize_hard_code(hc)
    
    # lsh
    lsh = LSH(num_bits=sp_model.model.hash_dim, num_tables=30, bits_per_hash=6)
    lsh.build_hash_tables(hc)
    approx_collision_matrix = lsh.export_adjacency_matrix()
    res = lsh.compare_adjacency_matrices(approx_collision_matrix,target_collision_matrix)
    print(e.n_sta**2,res)


    I = e.get_interfering_node_matrix()
    I = csr_matrix(I).astype(np.int8)
    I.eliminate_zeros()
    res = lsh.compare_adjacency_matrices(I,target_collision_matrix)
    print(e.n_sta**2,res)
