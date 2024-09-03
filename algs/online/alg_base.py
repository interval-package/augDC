from abc import abstractmethod
import copy
import numpy as np
from scipy.spatial import KDTree
import torch
import torch.nn.functional as F
from net.actor import Actor
from net.critic import DuelCritic
from typing import List, Tuple, Union, Literal
import gym
from algs import AlgBase
from utils.buffer import ReplayBuffer, OnlineSampler
from utils.np2t import Transition, np2tensor, tensor_trans, stack_trans, trans_obs

class AlgBaseOnline(AlgBase):

    def __init__(self, 
                env,
                state_dim,
                action_dim,
                max_action,
                device,
                discount=0.99,
                tau=0.005,
                policy_noise=0.2,
                noise_clip=0.5,
                policy_freq=2,
                actor_lr=3e-4,
                critic_lr=3e-4,
                alpha=2.5,
                **kwargs
                ) -> None:
        self.env = env
        self.device = torch.device(device)
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.critic = DuelCritic(state_dim, action_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha

        self.models = {
            "actor": self.actor,
            "critic": self.critic,
            "actor_optimizer": self.actor_optimizer,
            "critic_optimizer": self.critic_optimizer,
        }

        pass

    def train(
            self, 
            step:int, 
            rep_buffer: ReplayBuffer,
            sampler: OnlineSampler,
            mean=0,
            std=1,
            ):

        # A epi updarte once
        tb_info = {}
        info_grad = self._compute_gradient(data)
        info_upda = self._update(step)
        tb_info.update(info_grad)
        tb_info.update(info_upda)
        return tb_info

    @abstractmethod
    def _compute_gradient(self, data: Transition):
        ...

    @abstractmethod
    def _update(self, iteration):
        ...

    @property
    def adjustable_params(self):
        return  ("learning_rate")

    def close_critic_grad(self):
        for p in self.critic.parameters():
            p.requires_grad = False

    def close_actor_grad(self):
        for p in self.actor.parameters():
            p.requires_grad = False

    def open_critic_grad(self):
        for p in self.critic.parameters():
            p.requires_grad = True

    def open_actor_grad(self):
        for p in self.actor.parameters():
            p.requires_grad = True


    pass