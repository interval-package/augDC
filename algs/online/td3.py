from algs.online.alg_base import AlgBaseOnline, Transition
import copy
import torch
from net.actor import Actor
from net.critic import DuelCritic

class TD3(AlgBaseOnline):
    def __init__(self, 
                 gamma = 0.99,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.gamma = 0.99
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        for p in self.actor_target.parameters():
            p.requires_grad = False
        for p in self.critic_target.parameters():
            p.requires_grad = False

        models = {
            "actor_target": self.actor_target,
            "critic_target": self.critic_target,
        }
        self.models.update(models)

    def _compute_gradient(self, data: Transition):
        self.critic_optimizer.zero_grad()
        state, a, r, next_state, not_done = data
        loss_q, _, _ = self._compute_loss_q(state, a, r, next_state, not_done)
        loss_q.backward()

        self.close_critic_grad()
        self.actor_optimizer.zero_grad()
        loss_policy = self._compute_loss_policy(state)
        loss_policy.backward()
        self.open_critic_grad()
        return {}
    
    def _update(self, iteration):
        polyak = 1 - self.tau
        delay_update = self.policy_freq

        self.critic_optimizer.step()
        if iteration % delay_update == 0:
            self.actor_optimizer.step()

        # update target networks
        with torch.no_grad():
            for p, p_targ in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)
            for p, p_targ in zip(
                self.actor.parameters(),
                self.actor_target.parameters(),
            ):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)
        pass

    def _compute_loss_q(self, o, a, r, o2, d):
        q1, q2 = self.critic(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            pi_targ = self.actor(o2)
            # Target policy smoothing
            epsilon = torch.randn_like(pi_targ) * self.policy_noise
            epsilon = torch.clamp(epsilon, -self.noise_clip, self.noise_clip)
            a2 = pi_targ + epsilon
            a2 = torch.clamp(
                a2,
                torch.tensor(-self.max_action).to(a2.device),
                torch.tensor(self.max_action).to(a2.device),
            )

            # Target Q-values
            q1_pi_targ, q2_pi_targ = self.critic_target(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        return loss_q, loss_q1, loss_q2

    def _compute_loss_policy(self, o):
        q1_pi, q2_pi = self.critic(o, self.actor(o))
        return -q1_pi.mean()

    pass