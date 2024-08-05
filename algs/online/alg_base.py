from abc import abstractmethod
import copy
from scipy.spatial import KDTree
import torch
import torch.nn.functional as F
from net.actor import Actor
from net.critic import Critic
from typing import Tuple, Union, Literal

# state, action, reward, state_next, not_done
Transition = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,]

class AlgBase(object):

    def __init__(self, 
                env,
                actor:Actor, critic:Critic, max_action,
                device="cpu", 
                discount=0.99,
                tau=0.005,
                policy_noise=0.2,
                noise_clip=0.5,
                policy_freq=2,
                actor_lr=3e-4,
                critic_lr=3e-4,
                alpha=2.5,
                ) -> None:
        self.env = env
        self.device = torch.device(device)
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha
        # self.policy_target = copy.deepcopy(policy)
        # self.critic_target = copy.deepcopy(critic)
        pass

    def train(self):
        # TODO online exploration
        pass

    # TODO check the classmethod using the child class init or not
    @classmethod
    def from_offline(cls, env,off_alg):
        return cls(env, off_alg.actor, off_alg.critic, off_alg.max_action, off_alg.device)

    @classmethod
    def from_scratch(cls, 
        env,
        state_dim,
        action_dim,
        max_action,
        device,
        **kwargs
    ):
        device = torch.device(device)
        actor = Actor(state_dim, action_dim, max_action).to(device)
        critic = Critic(state_dim, action_dim).to(device)
        return cls(env, actor, critic, device)

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