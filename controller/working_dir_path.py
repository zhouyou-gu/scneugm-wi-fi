import os
from os.path import dirname, abspath

def get_working_dir_path():
    return dirname(abspath(__file__))

def get_ns3_path():
    path = os.path.abspath(os.path.join(dirname(abspath(__file__)), os.pardir))
    return os.path.join(path,"ns-3-dev")

def get_controller_path():
    return dirname(abspath(__file__))

if __name__ == '__main__':
    print(get_working_dir_path())
    print(get_ns3_path())