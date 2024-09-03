from typing import List, Tuple
import gym
import torch
import numpy as np

# state, action, reward, state_next, not_done
Transition = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,]

def np2tensor(tar, shape):
    if isinstance(tar, int):
        return torch.tensor([[tar]])
    if isinstance(tar, np.float64):
        return torch.tensor([[tar.astype(np.float32)]])
    elif isinstance(tar, np.ndarray):
        return torch.from_numpy(tar.reshape(shape).astype(np.float32))
    elif isinstance(tar, torch.Tensor):
        return tar
    else:
        raise TypeError(f"get invalid type: {type(tar)}")

def tensor_trans(data:Transition, env:gym.Env)->Transition:
    obs, action, rew, nobs, not_done = data
    action      = np2tensor(action,    (-1, env.action_space.shape[0]))
    rew         = np2tensor(rew,       (-1,1))
    nobs        = np2tensor(nobs,      (-1, env.observation_space.shape[0]))
    not_done    = np2tensor(not_done,  (-1,1))
    return (obs, action, rew, nobs, not_done)

def stack_trans(datas:List[Transition])->Transition:
    obs, action, rew, nobs, not_done = [torch.cat([data[i] for data in datas], dim=0) for i in range(5)]
    return (obs, action, rew, nobs, not_done)

def trans_obs(obs, mean, std):
    obs = (np.array(obs).reshape(1, -1) - mean) / std
    return obs
