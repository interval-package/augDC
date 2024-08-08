from algs.offline import AlgBaseOffline
import torch
import torch.nn.functional as F
from simulator.simulator_learn import simulator_base, simulator_learn
from net.actor import Actor
from net.critic import DuelCritic
from typing import Tuple
from copy import deepcopy

from utils.grad import get_network_grad

_datatuple = Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,]

class AugCDC(AlgBaseOffline):
    def __init__(self, data, simulator, **kwargs):
        super().__init__(data, **kwargs)
        self.simulator:simulator_learn = simulator

    @torch.no_grad()
    def augData_batch(self, init_state:torch.Tensor, actor:Actor, augBatch=256):
        """
        Using the actor virtually rollout N-step.
        Init state has batch dimension.
        WARNING not implemented well yet.
        """
        s1 = init_state
        l_s1, l_a, l_r, l_s2, l_nd = [], [], [], [], []
        for i in range(augBatch):
            act = actor(s1)
            noise = (torch.randn_like(act) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            act = (act + noise).clamp(
                -self.max_action, self.max_action
            )
            r, s2, nd = self.simulator.roll_out_step(s1, act)

            # Prepare for next round.
            s1 = s2 if nd else init_state

        def toTensor(ls:list):
            return torch.tensor(ls).unsqueeze(0)
        return toTensor(l_s1), toTensor(l_a), toTensor(l_r), toTensor(l_s2), toTensor(l_nd)

    @torch.no_grad()
    def augData(self, init_state:torch.Tensor, actor:Actor):
        """
        Aug data, because at batch level, 
        """
        action = actor(init_state)
        noise = (torch.randn_like(action) * self.policy_noise).clamp(
            -self.noise_clip, self.noise_clip
        )
        action = (action + noise).clamp(
            -self.max_action, self.max_action
        )
        reward, next_state, not_done = self.simulator.roll_out_step(init_state, action)
        return init_state, action, reward, next_state, not_done

    def augDataTuple(self, data:_datatuple, actor:Actor)->_datatuple:
        state, action, reward, next_state, not_done = data
        return self.augData(next_state, actor)

    def _calc_loss_actor(self, state:torch.Tensor):
        # Compute actor loss
        pi = self.actor(state)
        Q:torch.Tensor = self.critic.Q1(state, pi)
        lmbda = self.alpha / Q.abs().mean().detach()
        actor_loss = -lmbda * Q.mean()
        return actor_loss

    def _calc_nearest_neighbor_idx(self, state, pi):
        ## Get the nearest neighbor
        key = torch.cat([self.beta * state, pi], dim=1).detach().cpu().numpy()
        _, idx = self.kd_tree.query(key, k=[self.k], workers=-1)
        return idx

    def _calc_nearest_neighbor_idx_tuple(self, data:_datatuple)->Tuple[_datatuple, float, int]:
        state, action, reward, next_state, not_done = data
        key = torch.cat([self.beta * state, action], dim=1).detach().cpu().numpy()
        dis, idx = self.kd_tree.query(key, k=[self.k], workers=-1)
        nearest_neightbour = (
            torch.tensor(self.data[idx][:, :, -self.action_dim :])
            .squeeze(dim=1)
            .to(self.device)
        )
        return 

    def calc_actor_loss(self, data:_datatuple):
        state, action, reward, next_state, not_done = data
        # Compute actor loss
        pi = self.actor(state)
        Q:torch.Tensor = self.critic.Q1(state, pi)
        lmbda = self.alpha / Q.abs().mean().detach()
        actor_loss = -lmbda * Q.mean()
        return actor_loss, pi

    def train(self, replay_buffer, batch_size=256):
        # return super().train(replay_buffer, batch_size)
        self.total_it += 1
        tb_statics = dict()

        # Sample replay buffer
        # state, action, reward, next_state, not_done = replay_buffer.sample(batch_size)
        sampledTuple:_datatuple = replay_buffer.sample(batch_size)

        augTuple:_datatuple = self.augDataTuple(sampledTuple, self.actor)

        self.critic_optimizer.zero_grad()
        critic_loss_sampled = self._calc_loss_critic(sampledTuple)
        critic_loss_sampled.backward()
        grad_sampled = get_network_grad(self.critic)

        self.critic_optimizer.zero_grad()
        critic_loss_aug = self._calc_loss_critic(augTuple)
        critic_loss_aug.backward()
        grad_aug = get_network_grad(self.critic)

        self.critic_optimizer.zero_grad()

        # tb_statics.update({"critic_loss": critic_loss.item()})

        # Optimize the critic


        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # only using sampled data update actor
            state, action, reward, next_state, not_done = sampledTuple

            # Compute actor loss
            pi = self.actor(state)
            Q = self.critic.Q1(state, pi)
            lmbda = self.alpha / Q.abs().mean().detach()
            actor_loss = -lmbda * Q.mean()

            ## Get the nearest neighbor
            key = torch.cat([self.beta * state, pi], dim=1).detach().cpu().numpy()
            _, idx = self.kd_tree.query(key, k=[self.k], workers=-1)
            ## Calculate the regularization
            nearest_neightbour = (
                torch.tensor(self.data[idx][:, :, -self.action_dim :])
                .squeeze(dim=1)
                .to(self.device)
            )
            dc_loss = F.mse_loss(pi, nearest_neightbour)

            # Optimize the actor
            combined_loss = actor_loss + dc_loss
            self.actor_optimizer.zero_grad()
            combined_loss.backward()
            self.actor_optimizer.step()

            tb_statics.update(
                {
                    "dc_loss": dc_loss.item(),
                    "actor_loss": actor_loss.item(),
                    "combined_loss": combined_loss.item(),
                    "Q_value": torch.mean(Q).item(),
                    "lmbda": lmbda,
                }
            )

            self._update_target()

        return tb_statics

