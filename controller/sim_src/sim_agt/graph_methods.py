import numpy as np

from sim_src.sim_agt.sim_agt_base import sim_agt_base
from scipy.sparse import csr_matrix

class coloring_interference_matrix(sim_agt_base):
    def get_action(self):
        act = sim_agt_base.greedy_coloring(self.env.get_CH_matrix())
        return self.convert_action(act)


        
if __name__ == "__main__":
    from sim_src.sim_env.env import WiFiNet
    e = WiFiNet(seed=10)
    a = coloring_interference_matrix()
    a.set_env(e)
    print(a.get_action())


