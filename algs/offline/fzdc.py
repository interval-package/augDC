import copy
from scipy.spatial import KDTree
import torch
import torch.nn.functional as F
from algs.offline import AlgBaseOffline
from simulator import simulator_learn

"""
Code for feasible zone for dataset constrain
"""

class FZDC(AlgBaseOffline):
    def __init__(self, data_s, simulator, beta=2, k=1, **kwargs):
        super().__init__(data, **kwargs)
        self.beta=beta
        self.k = k
        self.kd_tree = KDTree(data)
        self.simulator:simulator_learn = simulator
        pass

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

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

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

            # Update the frozen target models
            self._update_target()

        return tb_statics