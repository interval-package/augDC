import copy
from scipy.spatial import KDTree
import torch
import torch.nn.functional as F
from algs.offline import AlgBaseOffline
from utils.np2t import Transition
from net.guard import DuelGuard

class PRWIC(AlgBaseOffline):
    def __init__(self, 
                 data, 
                 beta=2, 
                 k=1, 
                 guard_lr=0.01,
                 gamma=0.1,
                 **kwargs):
        super().__init__(**kwargs)
        self.data=data
        self.beta=beta
        self.k = k
        self.kd_tree = KDTree(data)
        self.gamma = gamma
        self.guard = DuelGuard(self.state_dim, self.action_dim).to(self.device)
        self.guard_target = copy.deepcopy(self.guard).to(self.device)
        self.guard_optimizer = torch.optim.Adam(self.guard.parameters(), lr=guard_lr)
        pass

    def _calc_loss_guard(self, trans_t:Transition):
        # Here we do not using the world model inference but by using the one step (s, a) dataset constrain to measure the constrain signal.
        state, action, reward, next_state, not_done = trans_t
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )

            pi = self.actor(next_state)
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
            p2d = torch.clamp(F.mse_loss(pi, nearest_neightbour) - 0.1, min=0, max=1)

            # Compute the target C value
            target_C1, target_C2 = self.guard_target(next_state, next_action)
            target_C = torch.min(target_C1, target_C2)
            # target_C = p2d + not_done * self.discount * target_C
            target_C = torch.max(p2d , target_C)

        current_C1, current_C2 = self.guard(state, action)

        # Compute guard loss
        guard_loss = F.mse_loss(current_C1, target_C) + F.mse_loss(
            current_C2, target_C
        )
        return guard_loss, nearest_neightbour

    def train(self, replay_buffer, batch_size=256):
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
        guard_loss, nearest_neightbour = self._calc_loss_guard((state, action, reward, next_state, not_done))
        tb_statics.update({"guard_loss": guard_loss.item()})

        # Optimize the guard
        self.guard_optimizer.zero_grad()
        guard_loss.backward()
        self.guard_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            pi = self.actor(state)
            Q = self.critic.Q1(state, pi)
            lmbda = self.alpha / Q.abs().mean().detach()
            actor_loss = -lmbda * Q.mean()


            # ## Get the nearest neighbor
            # key = torch.cat([self.beta * state, pi], dim=1).detach().cpu().numpy()
            # _, idx = self.kd_tree.query(key, k=[self.k], workers=-1)
            # ## Calculate the regularization
            # nearest_neightbour = (
            #     torch.tensor(self.data[idx][:, :, -self.action_dim :])
            #     .squeeze(dim=1)
            #     .to(self.device)
            # )
            dc_loss = F.mse_loss(pi, nearest_neightbour)

            C = self.guard.Q1(state, pi)
            cons_loss = C.mean() * self.gamma

            # Optimize the actor
            combined_loss = actor_loss + dc_loss + cons_loss
            self.actor_optimizer.zero_grad()
            combined_loss.backward()
            self.actor_optimizer.step()

            tb_statics.update(
                {
                    "dc_loss": dc_loss.item(),
                    "cons_loss": cons_loss.item(),
                    "actor_loss": actor_loss.item(),
                    "combined_loss": combined_loss.item(),
                    "Q_value": torch.mean(Q).item(),
                    "C_value": torch.mean(C).item(),
                    "lmbda": lmbda,
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