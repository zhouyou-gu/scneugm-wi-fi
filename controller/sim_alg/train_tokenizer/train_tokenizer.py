import numpy as np
from sim_mld.tokenizer.model import tokenizer_base
from sim_src.sim_env.env import WiFiNet

from sim_src.util import *

np.set_printoptions(precision=3)

model = tokenizer_base()

N_TRAINING_STEP = 2000
for i in range(N_TRAINING_STEP):
    e = WiFiNet(seed=i)
    b = e.get_sta_states()
    model.step(b)
    _ , r = model.get_output_np(b[0])
    print(r)
    print(b[0])

model.save(GET_LOG_PATH_FOR_SIM_SCRIPT(__file__),"final")
model.save_np(GET_LOG_PATH_FOR_SIM_SCRIPT(__file__),"final")
