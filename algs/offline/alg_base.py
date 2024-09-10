from abc import abstractmethod
import copy
from scipy.spatial import KDTree
import torch
import torch.nn.functional as F
from net.actor import Actor
from net.critic import DuelCritic
from typing import Tuple, Union, Literal
from algs import AlgBase
from utils.np2t import Transition

class AlgBaseOffline(AlgBase):
    def __init__(
        self,
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
        # beta=2,  # [beta* state, action]
        # k=1,
        **kwargs
    ):
        self.device = torch.device(device)
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic = DuelCritic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.state_dim =state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha

        # self.k = k
        self.total_it = 0
        # KD-Tree
        # self.beta = beta
        # self.data = data
        # self.kd_tree = KDTree(data)

        self.models = {
            "actor": self.actor,
            "critic": self.critic,
            "actor_target": self.actor_target,
            "critic_target": self.critic_target,
            "actor_optimizer": self.actor_optimizer,
            "critic_optimizer": self.critic_optimizer,
        }

        print("state_dim:", state_dim, ", action_dim: ", action_dim)

    @property
    def alg_params(self)->dict:
        temp = super().alg_params
        ret = {
            "type": "offline",
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "max_action": self.max_action,
            "discount": self.discount,
            "tau": self.tau,
            "policy_noise": self.policy_noise,
            "noise_clip": self.noise_clip,
            "policy_freq": self.policy_freq,
            "alpha": self.alpha,
        }
        temp.update(ret)
        return temp

    def _calc_loss_critic(self, 
                      trans_t:Transition):
        state, action, reward, next_state, not_done = trans_t
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )

            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )
        # tb_statics.update({"critic_loss": critic_loss.item()})

        return critic_loss
    
    def _update_target(self):
        """
        This method used to update the target networks.
        """
        # Update the frozen target models
        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        for param, target_param in zip(
            self.actor.parameters(), self.actor_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    @abstractmethod
    def train(self, **kwargs):
        ...