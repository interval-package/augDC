from algs.online.td3 import TD3
import gym
import torch
import torch.nn.functional as F
from scipy.spatial import KDTree

class TD3DC(TD3):
    def __init__(self, data, beta=2, k=1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.beta = beta
        self.k = k
        self.make_tree(data)

    def make_tree(self, data):
        self.data = data
        self.kd_tree = KDTree(data)
        pass

    def _compute_loss_policy(self, state):
        pi = self.actor(state)
        q1_pi, q2_pi = self.critic(state, pi)
        lmbda = self.alpha / q1_pi.abs().mean().detach()
        actor_loss = -lmbda * q1_pi.mean()

        key = torch.cat([self.beta * state, pi], dim=1).detach().cpu().numpy()
        _, idx = self.kd_tree.query(key, k=[self.k], workers=-1)
        nearest_neightbour = (
            torch.tensor(self.data[idx][:, :, -self.action_dim :])
            .squeeze(dim=1)
            .to(self.device)
        )

        dc_loss = F.mse_loss(pi, nearest_neightbour)

        info = {
            "lmbda": lmbda.item(),
            "actor_loss": actor_loss.item(),
            "dc_loss": dc_loss.item()
        }

        return actor_loss+dc_loss, info

    pass