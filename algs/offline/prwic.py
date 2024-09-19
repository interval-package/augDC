from abc import ABC, abstractmethod
import copy
from typing import Tuple
from scipy.spatial import KDTree
import torch
import torch.nn.functional as F
from algs.offline import AlgBaseOffline
from utils.np2t import Transition
from net.guard import DuelGuard

class PRWIC(AlgBaseOffline, ABC):
    def __init__(self, 
                 data, 
                 beta=2, 
                 k=1, 
                 guard_lr=0.01,
                 gamma_c=1e-4,
                 guard_factor=1e-2,
                 epsilon_c=1e-4,
                 max_timesteps=10000,
                 warm_factor_guard=0.4,
                 **kwargs):
        super().__init__(**kwargs)
        self.data=data
        self.beta=beta
        self.k = k
        self.kd_tree = KDTree(data)
        self.guard_factor = guard_factor
        self.gamma_c = gamma_c
        self.epsilon_c = epsilon_c
        self.max_timesteps = max_timesteps
        self.warm_up_step = int(warm_factor_guard * max_timesteps)

        self.guard = DuelGuard(self.state_dim, self.action_dim).to(self.device)
        self.guard_target = copy.deepcopy(self.guard).to(self.device)
        self.guard_optimizer = torch.optim.Adam(self.guard.parameters(), lr=guard_lr)
        pass

    @property
    def alg_params(self):
        ret = super().alg_params
        temp = {
            "balance_factor": self.guard_factor,
            "gamma_c": self.gamma_c,
            "epsilon": self.epsilon_c,
        }
        ret.update(temp)
        return ret

    @abstractmethod
    def _calc_target_c(self, p2d, target_C, not_done):
        ...

    def _calc_loss_guard(self, trans_t:Transition)->Tuple[torch.Tensor, dict]:
        # Here we do not using the world model inference but by using the one step (s, a) dataset constrain to measure the constrain signal.
        state, action, reward, next_state, not_done = trans_t
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )

            # pi_next = self.actor(next_state)
            pi = self.actor(state)
            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )

            key = torch.cat([self.beta * state, pi], dim=1).detach().cpu().numpy()
            _, idx = self.kd_tree.query(key, k=[self.k], workers=-1)
            ## Calculate the regularization
            nearest_neightbour = (
                torch.tensor(self.data[idx][:, :, -self.action_dim :])
                .squeeze(dim=1)
                .to(self.device)
            )

            # Change the constrain signal
            p2d = torch.clamp(F.mse_loss(pi, nearest_neightbour) - self.epsilon_c, min=0, max=0.5)

            # Compute the target C value
            target_C1, target_C2 = self.guard_target(next_state, next_action)
            target_C = torch.max(target_C1, target_C2)
            # target_C = p2d + not_done * self.discount * target_C
            target_C = self._calc_target_c(p2d, target_C, not_done)

        current_C1, current_C2 = self.guard(state, action)

        # Compute guard loss
        guard_loss = F.mse_loss(current_C1, target_C) + F.mse_loss(
            current_C2, target_C
        )

        info = {
            "pi": pi,
            "nearest_neightbour": nearest_neightbour, 
        }

        return guard_loss, info

    def train(self, replay_buffer, batch_size=256, **kwargs):
        self.total_it += 1
        tb_statics = dict()

        # Sample replay buffer
        state, action, reward, next_state, not_done = replay_buffer.sample(batch_size)

        # Compute critic loss
        critic_loss = self._calc_loss_critic((state, action, reward, next_state, not_done))
        tb_statics.update({"critic_loss": critic_loss.item()})

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute guard loss
        guard_loss, info = self._calc_loss_guard((state, action, reward, next_state, not_done))
        nearest_neightbour = info.pop("nearest_neightbour")
        tb_statics.update({"guard_loss": guard_loss.item()})

        # Optimize the guard
        self.guard_optimizer.zero_grad()
        guard_loss.backward()
        self.guard_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss, actually making it a BC
            pi = self.actor(state)
            Q = self.critic.Q1(state, pi)
            lmbda_Q = self.alpha / Q.abs().mean().detach()
            actor_loss = -lmbda_Q * Q.mean()

            dc_loss = F.mse_loss(pi, nearest_neightbour)

            if self.total_it > self.warm_up_step:
                # Trick Q like c loss calc, here different from above, treat like a true comp
                C = self.guard.Q1(state, pi)
                # lmbda_C = self.alpha / C.abs().mean().detach()
                # cons_loss = lmbda_C + C.mean() 
                cons_loss = C.mean()
                guard_factor = self.guard_factor
            else:
                cons_loss = C = torch.tensor(0).float()
                guard_factor = 0

            # Optimize the actor
            combined_loss = actor_loss + cons_loss * guard_factor + dc_loss * (1-guard_factor)
            self.actor_optimizer.zero_grad()
            combined_loss.backward()
            self.actor_optimizer.step()

            tb_statics.update(
                {
                    "dc_loss": dc_loss.item(),
                    "cons_loss": cons_loss.item(),
                    "dc_cons_gap": dc_loss.item() - cons_loss.item(),
                    "actor_loss": actor_loss.item(),
                    "combined_loss": combined_loss.item(),
                    "Q_value": torch.mean(Q).item(),
                    "C_value": torch.mean(C).item(),
                    "lmbda": lmbda_Q,
                }
            )

            # Update the frozen target models
            self._update_target()

        return tb_statics
        
    def _update_target(self):
        super()._update_target()
        for param, target_param in zip(
            self.guard.parameters(), self.guard_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
        
        return 


class PRWIC_max(PRWIC):
    def _calc_target_c(self, p2d, target_C, not_done):
        return torch.max(p2d , target_C * not_done) * self.gamma_c + p2d * (1-self.gamma_c)

class PRWIC_sum(PRWIC):
    def _calc_target_c(self, p2d, target_C, not_done):
        return p2d + not_done * self.gamma_c * target_C