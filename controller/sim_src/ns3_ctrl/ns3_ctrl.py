import os
import subprocess
import sys

VERBOSE = False

class ns3_env():
    def _run_ns3_proc(self):
        pass


def run_ns3(path_to_ns3, program_name, port, sim_seed = 1, sim_time = 10, sim_args = {}, debug=False):
    assert port is not None,  "run_ns3: need a specific port for ns3 program"

    cwd = os.getcwd()
    os.chdir(os.path.expandvars(path_to_ns3))
    ns3_string = "./ns3 run '" + program_name
    ns3_string += ' --openGymPort=' + str(port)
    ns3_string += ' --simSeed=' + str(sim_seed)
    ns3_string += ' --simTime=' + str(sim_time)

    for key, value in sim_args.items():
        ns3_string += " --"
        ns3_string += str(key)
        ns3_string += "="
        ns3_string += str(value)
    ns3_string += "' --no-build"
    out_d =  None if debug else subprocess.DEVNULL
    ns3_proc = subprocess.Popen(ns3_string, shell=True, stdout=out_d, stderr=None)
    if debug:
        print("Start command: ", ns3_string)
        print("Started ns3 simulation script, Process Id: ", ns3_proc.pid)

    # go back to my dir
    os.chdir(cwd)
    return ns3_proc


def build_ns3(path_to_ns3, debug=True):
    """
    Actually build the ns3 scenario before running.
    """
    cwd = os.getcwd()
    os.chdir(path_to_ns3)

    ns3_string = './ns3 build'

    output = subprocess.DEVNULL
    if debug:
        output = None

    build_required = False
    ns3_proc = subprocess.Popen(ns3_string, shell=True, stdout=subprocess.PIPE, stderr=None, universal_newlines=True)

    line_history = []
    for line in ns3_proc.stdout:
        if (True or "Compiling" in line or "Linking" in line) and not build_required:
            build_required = True
            print("Build ns-3 project if required")
            for l in line_history:
                sys.stdout.write(l)
                line_history = []

        if build_required:
            sys.stdout.write(line)
        else:
            line_history.append(line)

    p_status = ns3_proc.wait()
    if build_required:
        print("(Re-)Build of ns-3 finished with status: ", p_status)
    os.chdir(cwd)

def configure_ns3(path_to_ns3, debug=True):
    cwd = os.getcwd()
    os.chdir(path_to_ns3)

    ns3_string = './ns3 configure'

    output = subprocess.DEVNULL
    if debug:
        output = None

    ns3_proc = subprocess.Popen(ns3_string, shell=True, stdout=subprocess.PIPE, stderr=None, universal_newlines=True)

    for line in ns3_proc.stdout:
        sys.stdout.write(line)

    os.chdir(cwd)

def install_ns3gym(path_to_ns3, path_to_controller,debug=True):
    ns3gym_path = os.path.join(path_to_ns3,"contrib/opengym/model/ns3gym/ns3gym")
    ns3gym_path_to = os.path.join(path_to_controller)

    ns3_string = 'cp -r '+ ns3gym_path + ' ' + ns3gym_path_to

    output = subprocess.DEVNULL
    if debug:
        output = None

    ns3_proc = subprocess.Popen(ns3_string, shell=True, stdout=subprocess.PIPE, stderr=None, universal_newlines=True)

    for line in ns3_proc.stdout:
        sys.stdout.write(line)

def remove_ns3gym(path_to_controller,debug=True):
    ns3gym_path_to = os.path.join(path_to_controller,'ns3gym')

    ns3_string = 'rm -rf '+ ns3gym_path_to

    output = subprocess.DEVNULL
    if debug:
        output = None

    ns3_proc = subprocess.Popen(ns3_string, shell=True, stdout=subprocess.PIPE, stderr=None, universal_newlines=True)

    for line in ns3_proc.stdout:
        sys.stdout.write(line)
