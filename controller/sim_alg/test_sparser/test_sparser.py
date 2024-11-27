import numpy as np
import scipy
from scipy.sparse import csr_matrix

from sim_src.ns3_ctrl.sim_sys import sim_sys
from sim_src.sim_agt.sim_agt_base import agt_for_training
from working_dir_path import get_controller_path
from sim_mld.sparser.model import sparser_base
from sim_mld.sparser.lsh import LSH
from sim_mld.tokenizer.model import tokenizer_base
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



N_REPEAT = 1
i = np.arange(1, 16)
j = np.arange(1, 31)
I, J = np.meshgrid(i, j, indexing='ij')
pairs = np.column_stack([I.ravel(), J.ravel()])


for pair in pairs:
    n_bit, n_tab = pair
    print(f"Pair: ({n_bit}, {n_tab})")    
    for i in range(N_REPEAT):
        # get network state 
        def run():
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

            # lsh
            lsh = LSH(num_bits=hc.shape[1], num_tables=n_tab, bits_per_hash=n_bit)
            lsh.build_hash_tables(hc)
            approx_collision_matrix = lsh.export_adjacency_matrix()

            res = lsh.compare_adjacency_matrices(approx_collision_matrix,target_collision_matrix)
            print(res)



        import cProfile
        import pstats
        import io

        # Create a profiler
        pr = cProfile.Profile()
        pr.enable()  # Start profiling

        for i in range(1):
            run()
            
        pr.disable()  # Stop profiling

        # Create a string stream to capture the profiling data
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats(10)  # Print the top 10 functions

        # Output the profiling results
        print(s.getvalue())