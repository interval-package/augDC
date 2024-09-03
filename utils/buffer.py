import numpy as np
import torch
import d4rl


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, device, env_id: str, scale, shift, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.env_id = env_id

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device(device)
        self.scale = scale
        self.shift = shift 

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1.0 - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def clear(self):
        self.ptr = 0
        self.size = 0

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        transition =  [
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        ]
        
        # reward shaping, r = scale * r + shift, from CQL, FisherBRC, IQL, etc.
        # a common trick used for sparse reward env
        transition[2] = self.scale * transition[2] + self.shift
        
        return transition

    def convert_D4RL(self, dataset):
        self.state = dataset["observations"]
        self.action = dataset["actions"]
        self.next_state = dataset["next_observations"]
        self.reward = dataset["rewards"].reshape(-1, 1)
        self.not_done = 1.0 - dataset["terminals"].reshape(-1, 1)
        self.size = self.state.shape[0]

    def normalize_states(self, eps=1e-3):
        mean = self.state.mean(0, keepdims=True)
        std = self.state.std(0, keepdims=True) + eps
        self.state = (self.state - mean) / std
        self.next_state = (self.next_state - mean) / std
        return mean, std
    
    def from_d4rlenv(self, env):
        self.convert_D4RL(d4rl.qlearning_dataset(env))

from utils.np2t import Transition, np2tensor, tensor_trans, stack_trans, trans_obs

class OnlineSampler(ReplayBuffer):
    def __init__(self, mean=0, std=1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.mean=mean
        self.std=std

    def collect(self, alg, env, mean, std, batch, clear=False):
        if clear:
            self.clear()
        ptr = 0
        obs:np.ndarray = trans_obs(env.reset(), mean, std)
        obs = torch.from_numpy(obs.astype(np.float32)).to(self.device)
        not_done = 1
        while ptr<batch:
            while not_done:
                action: torch.Tensor = alg.actor(obs)
                noise = (torch.randn_like(action) * alg.policy_noise).clamp(
                    -alg.noise_clip, alg.noise_clip
                )
                action = (action + noise).clamp(
                    -alg.max_action, alg.max_action
                )
                _action = action.detach().numpy()
                nobs, rew, d, _ = env.step(_action)
                nobs = trans_obs(nobs, mean, std)
                not_done = 1 - d
                obs, action, rew, nobs, d = tensor_trans((obs, action, rew, nobs, d), env)
                ptr += 1
                self.add(obs, action, rew, nobs, d)
                
    pass