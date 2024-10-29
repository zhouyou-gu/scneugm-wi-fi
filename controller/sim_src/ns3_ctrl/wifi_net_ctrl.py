import subprocess
from threading import Thread

import numpy as np
from ns3gym import ns3env

from sim_src.ns3_ctrl.ns3_ctrl import run_ns3, ns3_env
from sim_src.util import STATS_OBJECT

import traceback

class wifi_net_config:
    PROG_PATH = ""
    PROG_NAME = ""
    def __init__(self):
        self.id = 0
            
        self.CMD_CONFIGS = {}
        self.CMD_CONFIGS["verbose"] = 0 # 0/1 for verbose enabled or disabled
        self.CMD_CONFIGS["MaxNumRetx"] = 5 # default, change here
        self.CMD_CONFIGS["phyMode"] = "OfdmRate6Mbps" # default, do not change
        self.CMD_CONFIGS["packetSize"] = 100 # updated in env
        self.CMD_CONFIGS["numPackets"] = 100 # updated in env
        self.CMD_CONFIGS["interval_in_us"] = 0  # not used 
        self.CMD_CONFIGS["n_ap"] = 0 # updated in env
        self.CMD_CONFIGS["n_sta"] = 0 # updated in env
        self.CMD_CONFIGS["openGymPort"] = 0 # automatically assigning a port number if 0
        self.CMD_CONFIGS["simSeed"] = 0 # updated in sim_sys
        self.CMD_CONFIGS["simTime"] = 0. # not used
        self.CMD_CONFIGS["TxPower"] = 5. # updated in env
        self.CMD_CONFIGS["RxNoiseFigure"] = 5. # updated in env
        self.CMD_CONFIGS["CcaEdThreshold"] = -95. # updated in env
        self.CMD_CONFIGS["RxSensitivity"] = -95. # updated in env
        self.CMD_CONFIGS["PreambleDetectionThresholdMinimumRssi"] = -95. # updated in env

        self.state = {}
        self.action = {}


class wifi_net_instance(STATS_OBJECT):
    def set_config(self, config):
        pass

    def get_return(self):
        pass

class sim_wifi_net(wifi_net_instance, ns3_env, Thread):
    DEBUG = False
    def __init__(self, id):
        Thread.__init__(self)
        self.id = id

        self.cfg:wifi_net_config = wifi_net_config()

        self.agnt = None
        self.proc = None

        self.ret = None

    def set_config(self, cfg:wifi_net_config):
        self.cfg = cfg

    def get_return(self):
        return self.ret

    def run(self):
        self._print("ns3 gym dt agent",self.id,"starts")
        self.agnt = ns3env.Ns3Env(port=self.cfg.CMD_CONFIGS["openGymPort"], startSim=False)
        self._print("ns3 sim",self.id,"starts")
        self.proc = self._run_ns3_proc()
        self.agnt.init()
        try:
            obs = self.agnt.reset()
            obs, reward, done, info = self.agnt.step(self._gen_ns3gym_act())
            self._ret_ns3gym_obs(obs)
        except Exception as e:
            print("sim_wifi_net run Error", str(e))
            traceback.print_exc()
        finally:
            self.agnt.close()
        self._print("ns3 gym dt agent",self.id,"is done")
        self.proc.wait()

    def _gen_ns3gym_act(self):
        act = {}
        #fill the path loss configuration
        act['loss_ap_ap'] = self.cfg.state['loss_ap_ap'].flatten().tolist()
        act['loss_sta_ap'] =  self.cfg.state['loss_sta_ap'].flatten().tolist()
        act['loss_sta_sta'] =  self.cfg.state['loss_sta_sta'].flatten().tolist()
        
        #fill the twt configuration
        act['twtstarttime'] =  self.cfg.action['twtstarttime'].flatten().tolist()
        act['twtoffset'] =  self.cfg.action['twtoffset'].flatten().tolist()
        act['twtduration'] =  self.cfg.action['twtduration'].flatten().tolist()
        act['twtperiodicity'] =  self.cfg.action['twtperiodicity'].flatten().tolist()
        return act

    def _ret_ns3gym_obs(self, obs):
        assert self.ret is None, "this ns3 instance already has return value"
        self.ret= {}
        for k in obs.keys():
            self.ret[k] = np.array(obs[k][:])

    def _run_ns3_proc(self) -> subprocess.Popen:
        path = self.cfg.PROG_PATH
        name = self.cfg.PROG_NAME
        if self.cfg.CMD_CONFIGS["openGymPort"] == 0:
            self.cfg.CMD_CONFIGS["openGymPort"] = self.agnt.get_port()
        return run_ns3(path_to_ns3=path, program_name=name, sim_args=self.cfg.CMD_CONFIGS, debug=self.DEBUG)
    
if __name__ == "__main__":
    print(str("true"))