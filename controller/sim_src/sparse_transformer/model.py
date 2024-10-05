import os.path

import numpy as np
import torch
from torch import optim

from sim_src.util import USE_CUDA, STATS_OBJECT, soft_update_inplace


class base_model(STATS_OBJECT):
    def __init__(self,  LR = 0.0001, TAU = 0.001, WITH_TARGET = False):
        self.WITH_TARGET = WITH_TARGET
        self.LR = LR
        self.TAU = TAU
        self.model = None
        self.model_target = None
        self.model_optim = None

        self.init_model()
        
        self.model_optim = optim.Adam(self.model.parameters(), lr=self.LR)

        self.move_model_to_gpu()
    
    def move_model_to_gpu(self):
        if USE_CUDA:
            self.model.to(torch.cuda.current_device())
            if self.WITH_TARGET:
                self.model_target.to(torch.cuda.current_device())
            print(self.__class__.__name__,"is on gpu")
        else:
            print(self.__class__.__name__,"is on cpu")

    def init_model(self):
        pass

    def load_model(self, path, path_target = None):
        if not path_target:
            path_target = path
        self.model = self._load(path)
        if self.WITH_TARGET:
            self.model_target = self._load(path_target)
        self.model_optim = optim.Adam(self.model.parameters(), lr=self.LR)

    def _load(self, path):
        if USE_CUDA:
            return torch.load(path, map_location=torch.device('cuda', torch.cuda.current_device()))
        else:
            return torch.load(path, map_location=torch.device('cpu'))
    
    def update_nn(self):
        if self.WITH_TARGET:
            soft_update_inplace(self.model_target, self.model, self.TAU)

    def save(self, path: str, postfix: str):
        try:
            os.mkdir(path)
        except:
            pass
        torch.save(self.model, os.path.join(path, self.__class__.__name__+ "." + postfix + ".pt"))
        if self.WITH_TARGET:
            torch.save(self.model_target, os.path.join(path, self.__class__.__name__+ "_target." + postfix + ".pt"))

    def step(self, batch):
        pass

    def get_output_np(self, input_np:np.ndarray)->np.ndarray:
        pass