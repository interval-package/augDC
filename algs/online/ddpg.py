from algs.online.alg_base import AlgBaseOnline, Transition
import copy
import torch
from net.actor import Actor
from net.critic import DuelCritic

# Warning the critic is the duel critic, not usable for DDPG
class DDPG(AlgBaseOnline):
    def __init__(self, **kwargs) -> None:
        raise NotImplementedError("The DDPG requires single critic.")
        super().__init__(**kwargs)
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
        loss_q, q = self._compute_loss_q(state, a, r, next_state, not_done)
        loss_q.backward()

        self.close_critic_grad()
        self.actor_optimizer.zero_grad()
        loss_policy = self._compute_loss_policy(state)
        loss_policy.backward()
        self.open_critic_grad()
        return
    
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

    def _compute_loss_q(self, state, action, rew, next_state, not_done):
        # Q-values
        Q = self.critic(state, action)

        # Target Q-values
        q_policy_targ = self.critic_target(next_state, self.actor_target(next_state))
        breakpoint()
        backup = rew + self.discount * not_done * q_policy_targ

        # MSE loss against Bellman backup
        loss_q = ((Q - backup) ** 2).mean()
        return loss_q, torch.mean(Q)

    def _compute_loss_policy(self, state):
        q_policy = self.critic(state, self.networks.policy(state))
        return -q_policy.mean()

    pass