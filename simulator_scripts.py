import os
import gym
import torch
import json
import pickle
import d4rl

from utils.buffer import ReplayBuffer
from utils.config import get_config, save_config

from simulator.model.model_base import model_MLP
from simulator.simulator_learn import get_simulator_path, simulator_base, simulator_learn

"""
This script is used to train and test a simulator.
"""

def train_model_MLP(env_id, env, obj_config, dict_config):
    path_model = get_simulator_path(env_id, "MLP")
    path_model_config = get_simulator_path(env_id, "MLP", "json")
    device = torch.device("cpu")

    state_dim  = dict_config["state_dim"]
    action_dim = dict_config["action_dim"]
    # max_action = kwargs["max_action"]

    sim = simulator_learn(env_id=obj_config.env_id, env=env, model_type="MLP")

    def eval_func(mdl:model_MLP):
        sim.set_model(mdl)
        info = sim.test_simulator()
        return info
    
    def save_func(mdl:model_MLP):
        sim.set_model(mdl)
        sim.save()
        return {}

    model_config = {
        "mlp_shape": [state_dim + action_dim, 256, 256, state_dim + 2],
        "lr": 0.01,
        "batch_size": 256,
        "epoches": 1e6,
        "eval_rounds": 100,
        "eval_func": eval_func,
        "save_func": save_func
    }

    model = model_MLP(device, **model_config)

    replay_buffer = ReplayBuffer(dict_config["state_dim"], dict_config["action_dim"], device, env_id, obj_config.scale, obj_config.shift)
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))

    model.train(replay_buffer, **model_config)
    model.save(path_model)

    with open(path_model_config, "wt") as f:
        json.dump(model_config, f)

    return model, model_config

def test_simulator_mlp(env:gym.Env, sim:simulator_learn, **kwargs):
    info = sim.test_simulator(pbar=True)
    print("==========================")
    for key in info.keys():
        print(f"{key}:\t{info[key]}")
    print("==========================")
    return

if __name__ == "__main__":
    args, env, kwargs = get_config("PRDC")
    train_model_MLP(args.env_id, env, args, kwargs)

    sim = simulator_learn(env_id=args.env_id, env=env, model_type="MLP")
    sim.load_model("MLP")
    test_simulator_mlp(env, sim)
    pass
