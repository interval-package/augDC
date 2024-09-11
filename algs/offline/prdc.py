import copy
from scipy.spatial import KDTree
import torch
import torch.nn.functional as F
from algs.offline import AlgBaseOffline

class PRDC(AlgBaseOffline):
    def __init__(self, data, beta=2, k=1, **kwargs):
        super().__init__(**kwargs)
        self.data=data
        self.beta=beta
        self.k = k
        self.kd_tree = KDTree(data)
        pass

    def train(self, replay_buffer, batch_size=256, **kwargs):
        self.total_it += 1
        tb_statics = dict()

        # Sample replay buffer
        state, action, reward, next_state, not_done = replay_buffer.sample(batch_size)

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

        return tb_statics