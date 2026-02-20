import os
import time
from multiprocessing import Process
from os.path import expanduser
from working_dir_path import *

CTRL_PATH = get_controller_path()
PATH = os.path.dirname(os.path.realpath(__file__))
TEST_LIST = [
# "test_online_static_channel_filtering_2_hg_chg.py",
"test_online_static_channel_filtering_3_hg_chg.py",
"test_online_static_channel_filtering_4_hg_chg.py"
]

CMD_LIST = []
for t in TEST_LIST:
    test_path = os.path.join(PATH,t)
    cmd = "PYTHONPATH=" + CTRL_PATH+ " python3 " + test_path
    CMD_LIST.append(cmd)


def run_cmd(cmd):
    os.system(cmd)

if __name__ == "__main__":
    for cmd in CMD_LIST:
        p = Process(target=run_cmd, args=(cmd,))
        p.start()
        time.sleep(1)
        p.join()
        time.sleep(1)
   