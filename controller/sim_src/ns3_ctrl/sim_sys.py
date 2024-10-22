from sim_src.ns3_ctrl.wifi_net_ctrl import sim_wifi_net, wifi_net_config
from sim_src.util import *


class sim_sys(STATS_OBJECT):
    def __init__(self, id=0):
        self.id = id
        self.ns3_env = None
    @counted
    def step(self, env, agt, run_ns3 = True, seed = None, sync=False)->dict:
        cfg = wifi_net_config()
        #seed is updated using sim_sys
        if seed:
            cfg.CMD_CONFIGS["simSeed"] = seed
        else:
            cfg.CMD_CONFIGS["simSeed"] = (self.N_STEP % 1000) + 5000
            
        cfg.state = env.get_state()
        cfg.action = agt.get_action()
        
        # env and agt needs to update the cfg of the system
        env_cfg:dict = env.get_config()        
        for k in env_cfg.keys():
            cfg.CMD_CONFIGS[k] = env_cfg[k]
            
        agt_cfg:dict = agt.get_config()        
        for k in agt_cfg.keys():
            cfg.CMD_CONFIGS[k] = agt_cfg[k]

        self.ns3_env = sim_wifi_net(self.id)
        self.ns3_env.set_config(cfg)

        if run_ns3:
            self.ns3_env.start()
            if sync:
                self.ns3_env.join()
                return self.process_ns3_return(self.ns3_env.get_return())
                
        return None
            
    def wait_ns3_end(self):
        self.ns3_env.join()
        return self.process_ns3_return(self.ns3_env.get_return())
    
    def is_ns3_end(self):
        return not self.ns3_env.is_alive()
    
    def process_ns3_return(self,ret):
        return ret['pkc']


if __name__ == "__main__":    
    from working_dir_path import get_ns3_path
    from sim_src.ns3_ctrl.ns3_ctrl import build_ns3
    build_ns3(get_ns3_path())
    
    from sim_src.sim_agt.graph_methods import coloring_interference_matrix
    from sim_src.sim_agt.sim_agt_base import sim_agt_base
    from sim_src.sim_env.env import WiFiNet
    wifi_net_config.PROG_PATH = get_ns3_path()
    wifi_net_config.PROG_NAME = "sparse-wi-fi/ns3gym_test/env"
    print("++++++++",wifi_net_config().PROG_NAME)
    env = WiFiNet()
    agt = sim_agt_base()
    agt.set_env(env=env)
    ns3sys = sim_sys()
    
    ns3sys.step(env=env,agt=agt)
