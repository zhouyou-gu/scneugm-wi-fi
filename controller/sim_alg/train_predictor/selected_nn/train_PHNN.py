import numpy as np
from working_dir_path import get_controller_path
from sim_mld.tokenizer.model import tokenizer_base
from sim_mld.predictor.model import PHNN
from sim_src.sim_env.env import WiFiNet
from sim_src.sim_agt.sim_agt_base import sim_agt_base, agt_for_training

from sim_src.util import *

np.set_printoptions(precision=3)

# load tokenizer model
tk_model = tokenizer_base()
path = get_controller_path()
path = os.path.join(path, "sim_alg/train_tokenizer/selected_nn/tokenizer_base.final.pt")
tk_model.load_model(path=path)
tk_model.eval()

model = PHNN()

N_TRAINING_STEP = 2000
for i in range(N_TRAINING_STEP):
    env = WiFiNet(seed=i)
    agt = agt_for_training()

    # tokenize sta states
    b = env.get_sta_states()
    l, _ = tk_model.get_output_np_batch(b)
    H, edge_index = agt.export_all_edges(env.get_hidden_node_matrix())

    batch = {}
    batch["token"] = l
    batch["edge_index"] = edge_index
    batch["target"] = H
    model.step(batch)

model.save(GET_LOG_PATH_FOR_SIM_SCRIPT(__file__),"final")
model.save_np(GET_LOG_PATH_FOR_SIM_SCRIPT(__file__),"final")
