<h1 align="center">✨ScNeuGM-WI-FI✨</h1>

Source Codes for our paper, "[ScNeuGM: Scalable Neural Graph Modeling for Coloring-Based Contention and Interference Management in Wi-Fi 7](https://arxiv.org/pdf/2502.03300)," authored by Zhouyou Gu, Jihong Park, Jinho Choi.
![system](contention_and_interference.png)
Reference
```bibtex
@article{gu2025scneugm,
  title={ScNeuGM: Scalable Neural Graph Modeling for Coloring-Based Contention and Interference Management in Wi-Fi 7},
  author={Gu, Zhouyou and Park, Jihong and Choi, Jinho},
  journal={arXiv preprint arXiv:2502.03300},
  year={2025}
}
```

## Installation
Clone this repo and run the following to fetch the [NS-3 codes](https://github.com/zhouyou-gu/ns-3-dev-ac-grl-wi-fi/tree/ns3-sparse-wi-fi) for this project.
```bash
git submodule update --init --recursive
```

Install [Pytorch](https://pytorch.org/get-started/locally/) and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) and [NVIDIA Drivers](https://ubuntu.com/server/docs/nvidia-drivers-installation)

Install python packages
```
pip3 install -r requirements.txt
```

### NS-3 Related Installation

On Ubuntu (or WSL on Windows), install the ns3 and ns3gym dependency as (double check "protoc --version" and make the version match the python protobuf version). If not, the communication between ns-3 and algorithms will not be connected.
```bash
sudo apt-get update
sudo apt-get install gcc g++ python3 python3-pip cmake ninja-build ccache
sudo apt-get install libzmq5 libzmq3-dev
sudo apt-get install libprotobuf-dev
sudo apt-get install protobuf-compiler
sudo apt-get install pkg-config
```

On MAC, install the ns3 and ns3gym dependency as (env parameters are needed to be re-exported for a new terminal).
```bash
brew install cmake gcc pkg-config protobuf protobuf-c ninja zeromq cppzmq ccache

ls -l /opt/homebrew/lib/libprotobuf.dylib
ls -l /opt/homebrew/lib/libzmq.dylib
```

<mark>If the modification on ns-3 submodule is needed, ensure that the submodule is checked out to a branch other than a detached head; otherwise, the modification will be tracked by git.<mark>

### Compiling NS-3

<mark>Change the working directory to [controller](controller)</mark>


(Optional) Now, you can have a test on whether the codes are correctly connected to the torch installation as 
```bash
PYTHONPATH=./ python3 util_script/cuda_test.py
```
or
```bash
PYTHONPATH=./ python3 util_script/apple_test.py
```
(Optional) an example return looks like the following, but it depends on your system.
```
torch.version 1.12.0+cu116
torch.cuda.is_available() True
torch.cuda.current_device() 0
torch_geometric.__version__ 2.1.0
```

(On MAC) Before the compilation of ns-3, MAC OS needs the following env parameters.

```bash
export CPATH="/opt/homebrew/include:$CPATH"
export CPLUS_INCLUDE_PATH="/opt/homebrew/include:$CPLUS_INCLUDE_PATH"
export LDFLAGS="-L/opt/homebrew/lib $LDFLAGS"
export CPPFLAGS="-I/opt/homebrew/include $CPPFLAGS"
export PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig:$PKG_CONFIG_PATH"
export CMAKE_PREFIX_PATH="/opt/homebrew:$CMAKE_PREFIX_PATH"
export LIBRARY_PATH="/opt/homebrew/lib:$LIBRARY_PATH"
```

Next, configure and compile NS-3
```bash
PYTHONPATH=./ python3 util_script/configure_ns3.py
```

```bash
PYTHONPATH=./ python3 util_script/build_ns3.py
```

```bash
PYTHONPATH=./ python3 util_script/install_ns3gym.py
```

## Reproducing Results in the Paper
All Results in the paper can be reproduced using the scripts in [controller/sim_alg](controller/sim_alg)

The terminology in the codes and the paper are slightly different. Some mapping between the variables in codes and the terms in the paper are listed as follows. GGM<=>Neural Graph Modeling, Tokenizer<=>State Embedding, Sparser<=>Deep Hashing Function.

