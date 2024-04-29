from abc import ABC, abstractmethod
import json
import gym

path_config = "simulator/simulator_buffer/config.json"

class simulator_base:
    def __init__(self, env_id:str, env:gym.Env) -> None:
        self.env_id = env_id
        self.env = env
        pass

    def load_config(self):
        with open(path_config, "rt")  as f:
            config = json.load(f)
        return config

    @abstractmethod
    def roll_out_step(self, state, action):
        ...

    @abstractmethod
    def roll_out_traj(self, init_state, policy, length=5):
        ...
    
    pass