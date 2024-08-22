import os.path
from os.path import expanduser

from sim_src.ns3_ctrl.ns3_ctrl import install_ns3gym, remove_ns3gym
from working_dir_path import *

remove_ns3gym(get_controller_path())
install_ns3gym(get_ns3_path(),get_controller_path())