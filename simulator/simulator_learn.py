import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import os
import gym
import tqdm
from typing import Literal, Union
from simulator.simulator_base import simulator_base
import world_model.model as models
import time

"""
The simulator_learn is used to warp models to simulator
"""

path_script = os.path.abspath(__file__)

path_folder = os.path.dirname(path_script)

path_model_folder = os.path.join(path_folder, "..", "result", "world_models")

def get_simulator_folder(env_id, model_type, ftime:str=None):
    ftime = ftime if ftime is not None else time.strftime("%m-%d-%H:%M:%S")
    ret = os.path.join(path_model_folder, env_id, model_type, ftime)
    if not os.path.exists(ret):
        os.makedirs(ret)
    return ret

def calc_cosine_similarity(vector1, vector2):
    # Compute the dot product
    dot_product = np.dot(vector1, vector2)

    # Compute the magnitudes
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    # Compute the cosine similarity
    cosine_similarity = dot_product / (magnitude1 * magnitude2)
    return cosine_similarity

class simulator_learn(simulator_base):

    def __init__(self, env_id, env:gym.Env, model_type:Literal["MLP", "GAN", "VAE"], model_config=None, path=None, **kwargs):
        """
        Using the env id to specifize the target simulator.
        Check the buffer contains the simulator or not. If contains load the simulator, else train with d4rl data and save. 
        """
        super().__init__(env_id, env)
        self.model_type = model_type
        self.model_config = model_config

        self.env_model:models.model_base = None
        # self.load_model(env_id, model_type)
        self.path = path
        self.shapes = None

    def make_path(self):
        self.path = self.path if self.path is not None else get_simulator_folder(self.env_id, self.model_type)
        self.path_model = os.path.join(self.path, "models")
        if not os.path.exists(self.path_model):
            os.makedirs(self.path_model)

    def from_folder(cls, path=None, **kwargs):
        if path is None:
            path
            pass
        return cls

    @classmethod
    def from_model(cls, model, env, model_config:dict):
        obj = cls(model_config["env_id"], env, model_config["model_type"], model_config=model_config)
        obj.set_model(model)
        return obj

    def load_model(self, model_type):
        if self.env_model is not None:
            return self.env_model
        
        if os.path.exists(self.path_model):
            print(f"Load model from {self.path_model}.")
            with open(self.path_model, "rb") as f:
                ret = pickle.load(f)
            self.env_model = ret
        else:
            print(f"Model not exists at {self.path_model}")
            raise NotImplementedError("Do not allow empty load yet.")
            self.env_model:models.model_base = getattr(models, f"model_{model_type}")(**self.model_config)
            self.env_model.train()
            self.save()
        assert isinstance(self.env_model, models.model_base), f"Not a model but a {type(self.env_model)}"
        return self.env_model

    def set_model(self, model):
        self.env_model = model
        return model

    def test_simulator(self, eval_round=256, info=None, pbar=False, max_step=100,**kwargs):
        """
        This function is used to test the difference between the simulator and real env.
        Only consider one step rollout error. Using KL divergence.
        Return eval dict.
        """

        device = self.env_model.device

        if pbar:
            pbar = tqdm.tqdm(range(int(eval_round)))
            pbar.set_description("Test model consistency...")
        else:
            pbar = range(int(eval_round))
        r_acc, s_acc, d_acc = 0, 0, 0
        
        def reset():
            s_init:np.ndarray = self.env.reset()
            s_init:torch.Tensor = torch.from_numpy(s_init.astype(np.float32)).unsqueeze(0).to(device)
            return s_init
        
        s_init = reset()
        
        tt_step = 0

        for iter in pbar:
            # init_s = self.env.observation_space.sample()
            step = 0
            for i in range(max_step):
                act:np.ndarray = self.env.action_space.sample()
                s_env, r_env, d_env, _ = self.env.step(act)
                act:torch.Tensor = torch.from_numpy(act.astype(np.float32)).unsqueeze(0).to(device)
                r_mdl, s_mdl, d_mdl, _ = self.roll_out_step(s_init, act)
                r_acc = r_acc + np.abs(r_mdl.detach().cpu().numpy() - r_env)
                # cosine similarity
                cos_sim = np.abs(calc_cosine_similarity(s_mdl.detach().cpu().numpy(), s_env))
                s_acc = s_acc + cos_sim
                d_acc = d_acc + np.abs(d_mdl.detach().cpu().numpy() - d_env)
                step += 1
                if d_env:
                    s_init = reset()
                    break
                else:
                    s_init = torch.from_numpy(s_env.astype(np.float32)).unsqueeze(0).to(device)

            tt_step += step

        eval_dict = {
            "r_acc_mean": r_acc/eval_round,
            # "r_acc_std":  r_acc/eval_round,
            "s_acc_mean": s_acc/eval_round,
            # "s_acc_std":  s_acc/eval_round,
            "d_acc_mean": d_acc/eval_round,
            # "d_acc_std":  d_acc/eval_round,
            "avg_epi_len": tt_step/eval_round,
            "eval_round": eval_round,
        }
        return eval_dict

    @torch.no_grad()
    def roll_out_step(self, state, action):
        """
        This function should allow batch level forward.
        [batch, states].
        Return reward, n_state, done, information.
        """
        input = torch.cat([state, action], 1)
        output:torch.Tensor = self.env_model.forward(input)
        # reward, n_state, done = output[:, 1], output[:, 1:-1], output[:, -1]
        reward, n_state, done = output
        return reward, n_state, done, {}

    @torch.no_grad()
    def roll_out_traj(self, init_state, policy, length=5):
        state = init_state
        ret = []
        for i in range(length):
            action = policy(state)
            r, state, d, _ = self.roll_out_step(state, action)
            ret.append([r, state, d])
        return ret

    def save(self, iter=None):
        path_model = os.path.join(self.path_model, f"model_{iter if iter is not None else 'latest'}.pkl")
        with open(path_model, "wb") as f:
            pickle.dump(self.env_model, f)