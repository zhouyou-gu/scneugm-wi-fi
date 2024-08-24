import os
import random
from os.path import expanduser

import numpy as np

from sim_src.sim_env.sim import sim_env
from sim_src.util import GET_LOG_PATH_FOR_SIM_SCRIPT, STATS_OBJECT

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)

OUT_FOLDER = GET_LOG_PATH_FOR_SIM_SCRIPT(__file__)
STATS_OBJECT.DISABLE_ALL_DEBUG = False

class tmp_sim_env(sim_env):
    def format_act_to_sta_twt_idx(self, action):
        return action


e = tmp_sim_env(id=random.randint(40,60))
from working_dir_path import *
ns3_path = get_ns3_path()
e.PROG_PATH = ns3_path
e.PROG_NAME = "wifi-ai/env"
e.DEBUG = True

n_step = 1000

class test_actor:
    def __init__(self, logk = 2):
        self.logk = logk

    def gen_action(self,state_np):
        n_node = state_np.shape[0]
        ret = np.zeros(n_node)
        min_loss_ap_idx = np.argmin(state_np,axis=1)
        # total_counter = 0
        # for a in range(state_np.shape[1]):
        #     for s in range(n_node):
        #         min_a = min_loss_ap_idx[s]
        #         if min_a == a:
        #             ret[s] = total_counter % (2**self.logk)
        #             total_counter += 1
        # print(total_counter)
        # print(min_loss_ap_idx)
        # print(per_ap_counter)
        # print(ret)
        return min_loss_ap_idx


e.set_actor(test_actor(e.twt_log2_n_slot))
for i in range(n_step):
    e.init_env()
    sample = e.step(run_ns3=True)
    if (i+1) in [10, 100, 500, 1000]:
        e.save_np(OUT_FOLDER,str(i))