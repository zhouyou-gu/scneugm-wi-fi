import random

import numpy as np

from sim_src.sim_env.sim import sim_env

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)

# exit(0)
class tmp_sim_env(sim_env):
    def format_act_to_sta_twt_idx(self, action):
        return action


e = tmp_sim_env(id=random.randint(40,60))
e.PROG_PATH = "/home/soyo/wifi-ai/ns-3-dev"
e.PROG_NAME = "wifi-ai/env"
e.DEBUG = True

n_step = 100000

class test_actor:
    def __init__(self, n_sta, logk = 1):
        self.n_sta = n_sta
        self.logk = logk

    def gen_action(self,state):
        ret = np.random.randint(0,2**self.logk,self.n_sta)
        return ret

e.set_actor(test_actor(e.set_n_sta(), e.twt_log2_n_slot))
for i in range(n_step):
    e.init_env()
    sample = e.step(run_ns3=True)
