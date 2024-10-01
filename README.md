# SPARSE-GGM-WI-FI

```bash
sudo apt-get update
sudo apt-get install gcc g++ python3 python3-pip cmake
sudo apt-get install libzmq5 libzmq3-dev
sudo apt-get install libprotobuf-dev
sudo apt-get install protobuf-compiler
sudo apt-get install pkg-config

```


### Overall Code Structure

![code](ac-grl-wi-fi-code-structure.png)

### Installation

In your home directory, git clone this repository as (if you clone the repository in a different place, you may need to change [the path configuration](controller/working_dir_path.py) in this project)
```bash
git clone https://github.com/zhouyou-gu/ac-grl-wi-fi.git
```

Change the working directory to the cloned ac-grl-wi-fi folder. 
```bash
cd ac-grl-wi-fi
```
Run the following to fetch the [NS-3 codes](https://github.com/zhouyou-gu/ns-3-dev-ac-grl-wi-fi.git) for this project.
```bash
git submodule update --init --recursive
```

Change the working directory to [controller](controller)



(Optional) Now, you can have a test on whether the codes are correctly connected to the torch installation as 
```bash
PYTHONPATH=./ python3 sim_script/cuda_test.py
```
(Optional) for example, on the computer that we used to generate the simulation results, the above command returns the versions of torch as
```
torch.version 1.12.0+cu116
torch.cuda.is_available() True
torch.cuda.current_device() 0
torch_geometric.__version__ 2.1.0
```

Next, configure and compile NS-3
```bash
PYTHONPATH=./ python3 sim_script/configure_ns3.py
```
```bash
PYTHONPATH=./ python3 sim_script/build_ns3.py
```

To install ns3gym in Python, change the directory back to the repository root 
and then to [ns-3-dev/contrib/opengym](https://github.com/zhouyou-gu/ns-3-dev-ac-grl-wi-fi/tree/master/contrib/opengym), run

```bash
sudo pip3 install ./model/ns3gym    
```

### Simulations
Set the working directory as [controller](controller).

Run the pre-training of the inference NN in AC-GRL as
```bash
PYTHONPATH=./ python3 sim_src/algorithms/test_model/test_infer.py
```

Run the main training process of AC-GRL as
```bash
PYTHONPATH=./ python3 sim_src/algorithms/test_model/test_itl_mmf_bidirection_interference.py
```

Alternatively, the trained NN in the paper can be evaluated (alone with other compared schemes) as
```bash
PYTHONPATH=./ python3 sim_src/algorithms/eval_model/do_run_test_parallel.py
```

The evaluation of the online fine-tuning architecture is  
```bash
PYTHONPATH=./ python3 sim_src/algorithms/online_finetune/do_run_test_parallel.py
```
