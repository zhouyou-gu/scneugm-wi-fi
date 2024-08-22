import os
import random
from os.path import expanduser

import numpy as np
import torch

from sim_src.other_methods.graphcut_model import graphcut_hidden_model
from sim_src.sim_env.sim_env import sim_env
from sim_src.util import GET_LOG_PATH_FOR_SIM_SCRIPT, ParameterConfig, STATS_OBJECT

np.set_printoptions(threshold=5)
np.set_printoptions(linewidth=1000)

torch.set_printoptions(threshold=5)
torch.set_printoptions(linewidth=1000)

STATS_OBJECT.DISABLE_ALL_DEBUG = False


OUT_FOLDER = GET_LOG_PATH_FOR_SIM_SCRIPT(__file__)

e = sim_env(id=random.randint(40,200))

from working_dir_path import *
ns3_path = get_ns3_path()

e.PROG_PATH = ns3_path
e.PROG_NAME = "wifi-ai/env"
e.DEBUG = True


n_step = 1000
model = graphcut_hidden_model(0)
model.DEBUG_STEP = 10
model.DEBUG = True

cfg = ParameterConfig()
# cfg['ALPHA'] = ALPHA
# model.FAIRNESS_ALPHA = cfg['ALPHA']
cfg.save(OUT_FOLDER,"NaN")

e.set_actor(model)
for i in range(n_step):
    e.init_env()
    sample = e.step(run_ns3=True)
    if (i+1) in [10, 100, 500, 1000]:
        e.save_np(OUT_FOLDER,str(i))