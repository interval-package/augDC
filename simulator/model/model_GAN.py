from copy import deepcopy
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os
import tqdm
from typing import Callable, Dict, Literal, Tuple, Union
from abc import ABC, abstractmethod
from utils.buffer import ReplayBuffer

class model_GAN(nn.Module, model_base):
    def __init__(self) -> None:
        super.__init__()

    def train(self, replay_buffer):
        pass