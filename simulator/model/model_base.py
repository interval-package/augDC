import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os
import tqdm
from typing import Callable, Literal, Union
from abc import ABC, abstractmethod
from utils.buffer import ReplayBuffer

class model_base(ABC):

    @abstractmethod
    def train(self):
        ...

    @abstractmethod
    def forward(self, input):
        ...

    # @abstractmethod
    # def step(self, state, action):
    #     ...

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return

    pass

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)

class model_MLP(nn.Module, model_base):
    """
    A network to simulate the env model.
    """
    def __init__(self, device, mlp_shape, lr=0.01, batch_size=256, **kwargs) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.device = torch.device(device)
        
        self.model = mlp(mlp_shape, nn.ReLU)
        
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_f = nn.BCELoss()

    def forward(self, input:torch.Tensor):
        # do not clip output
        return self.model(input)

    def train(self, replay_buffer:ReplayBuffer, epoches=1e3, 
                eval_round:int=1000, eval_func:Callable[[model_base, int], Union[dict, str]]=None,
                save_round:int=1000, save_func:Callable[[model_base, int], Union[dict, str]]=None,
                **kwargs):
        eval_round = int(eval_round)
        save_round = int(save_round)
        pbar = tqdm.tqdm(range(int(epoches)))
        pbar.set_description("Train MLP....")
        for epoch in pbar:
            state, action, reward, next_state, not_done = replay_buffer.sample(self.batch_size)

            cur_real = torch.cat([state, action], 1)
            next_real = torch.cat([reward, next_state, not_done], 1)
            next_pred = self.forward(cur_real)

            loss = self.loss_f(torch.sigmoid(next_real), torch.sigmoid(next_pred))
            loss.backward()
            self.optimizer.step()

            if epoch % eval_round == 0:
                # default eval method
                if eval_func is not None:
                    msg = eval_func(self, epoch)
                    # print(msg)
                pass

            if epoch % save_round == 0:
                # default save method
                if save_func is not None:
                    msg = save_func(self, epoch)
                    # print(msg)
                pass
            pass
        
        print("Train finished.")

class model_GAN(nn.Module, model_base):
    def __init__(self) -> None:
        super.__init__()

    def train(self, replay_buffer):
        pass

if __name__ == "__main__":
    pass