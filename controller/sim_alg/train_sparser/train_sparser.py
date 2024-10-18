import numpy as np
from working_dir_path import get_controller_path
from model.sparse_transformer.sparser.model import sparser_base
from model.sparse_transformer.tokenizer.model import tokenizer_base
from sim_src.sim_env.env import WiFiNet

from sim_src.util import *

np.set_printoptions(precision=3)

# load tokenizer model
tk_model = tokenizer_base()
path = get_controller_path()
path = os.path.join(path, "sim_alg/train_tokenizer/selected_nn/tokenizer_base.final.pt")
tk_model.load_model(path=path)
tk_model.eval()

# setup sparser model
sp_model = sparser_base()


N_TRAINING_STEP = 10000
for i in range(N_TRAINING_STEP):
    # get network state 
    e = WiFiNet(seed=i)
    b = e.get_sta_states()
    
    # tokenize sta states
    l, _ = tk_model.get_output_np_batch(b)
    
    # get collision matrix
    target_collision_matrix = e.get_CH_matrix()
    
    # get batch
    batch = (l,target_collision_matrix)
    
    # step
    sp_model.step(batch=batch)


sp_model.save(GET_LOG_PATH_FOR_SIM_SCRIPT(__file__),"final")
sp_model.save_np(GET_LOG_PATH_FOR_SIM_SCRIPT(__file__),"final")
