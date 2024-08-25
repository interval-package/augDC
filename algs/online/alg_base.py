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

# state, action, reward, state_next, not_done
Transition = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,]

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

    @staticmethod
    def np2tensor(tar, shape):
        if isinstance(tar, int):
            return torch.tensor([[tar]])
        if isinstance(tar, np.float64):
            return torch.tensor([[tar.astype(np.float32)]])
        elif isinstance(tar, np.ndarray):
            return torch.from_numpy(tar.reshape(shape).astype(np.float32))
        elif isinstance(tar, torch.Tensor):
            return tar
        else:
            raise TypeError(f"get invalid type: {type(tar)}")

    def tensor_trans(self, data:Transition, env:gym.Env)->Transition:
        obs, action, rew, nobs, not_done = data
        action      = self.np2tensor(action,    (-1, env.action_space.shape[0]))
        rew         = self.np2tensor(rew,       (-1,1))
        nobs        = self.np2tensor(nobs,      (-1, env.observation_space.shape[0]))
        not_done    = self.np2tensor(not_done,  (-1,1))
        return (obs, action, rew, nobs, not_done)

    @staticmethod
    def stack_trans(datas:List[Transition])->Transition:
        obs, action, rew, nobs, not_done = [torch.cat([data[i] for data in datas], dim=0) for i in range(5)]
        return (obs, action, rew, nobs, not_done)

    @staticmethod
    def trans_obs(obs, mean, std):
        obs = (np.array(obs).reshape(1, -1) - mean) / std
        return obs

    def train(
            self, 
            env:gym.Env,
            step:int, 
            mean=0,
            std=1,
            max_epi_len=100
            ):

        return

    def _train(
            self, 
            env:gym.Env,
            step:int, 
            mean=0,
            std=1,
            max_epi_len=100
            ):
        """
        Train a episode.
        """
        # Assert here env outputs done
        obs:np.ndarray = self.trans_obs(env.reset(), mean, std)
        obs = torch.from_numpy(obs.astype(np.float32)).to(self.device)
        t = 0
        not_done = 1
        tt_rew = 0
        datas = []
        while not_done and t < max_epi_len:
            action: torch.Tensor = self.actor(obs)
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            action = (action + noise).clamp(
                -self.max_action, self.max_action
            )
            _action = action.detach().numpy()
            nobs, rew, d, _ = env.step(_action)
            nobs = self.trans_obs(nobs, mean, std)
            not_done = 1 - d
            data = self.tensor_trans((obs, action, rew, nobs, not_done), env)
            datas.append(data)
            tt_rew += rew
            t += 1
            obs = data[3]

        _data = self.stack_trans(datas=datas)
        # A epi updarte once
        info_grad = self._compute_gradient(_data)
        info_upda = self._update(step)
        tb_info = {
            "online/epi_len": t,
            "online/total_rew": tt_rew
        }
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