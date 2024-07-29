import os
import gym
import torch
import json
import pickle
import d4rl

from utils.buffer import ReplayBuffer
from utils.config import get_config, save_config
from utils.logger import add_scalars

from torch.utils.tensorboard import SummaryWriter

from simulator.model.model_base import model_MLP
from simulator.simulator_learn import get_simulator_path, get_simulator_folder, simulator_base, simulator_learn

"""
This script is used to train and test a simulator.
"""

def save_model_config(model_config:dict, path_model_config):
    ret = {}
    for key, value in model_config.items():
        if callable(value):
            v = value.__name__
        else:
            v = value
        ret[key] = v
        pass
    with open(path_model_config, "wt") as f:
        json.dump(ret, f)
    return

def train_model_MLP(env_id, env, obj_config, dict_config):
    model_type="MLP"
    path_folder = get_simulator_folder(env_id, "MLP")
    path_model_config = get_simulator_path(env_id, model_type, ftype="json")
    device = torch.device("cpu")

    state_dim  = dict_config["state_dim"]
    action_dim = dict_config["action_dim"]
    # max_action = kwargs["max_action"]

    writer = SummaryWriter(log_dir=path_folder, flush_secs=120)

    sim = simulator_learn(env_id=obj_config.env_id, env=env, model_type=model_type)

    def eval_func(mdl:model_MLP, iter:int, **kwargs):
        sim.set_model(mdl)
        info = sim.test_simulator(info={"iter":iter})
        if "loss" in kwargs.keys():
            info["loss"] = kwargs["loss"]
        add_scalars(info, writer=writer, step=iter)
        return info
    
    def save_func(mdl:model_MLP, iter:int, **kwargs):
        sim.set_model(mdl)
        sim.save(iter)
        return {"msg": f"Saved f{iter}."}

    model_config = {
        "mlp_shape": [state_dim + action_dim, 256, 256, state_dim + 2],
        "lr": 0.01,
        "batch_size": 256,
        "epoches": 1e7,
        "eval_round": 2000,
        "save_round": 100000,
        "eval_func": eval_func,
        "save_func": save_func
    }

    model = model_MLP(device, **model_config)

    replay_buffer = ReplayBuffer(dict_config["state_dim"], dict_config["action_dim"], device, env_id, obj_config.scale, obj_config.shift)
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))

    model.train(replay_buffer, **model_config)
    # model.save(path_model)

    save_model_config(model_config, path_model_config)

    return model, model_config

def test_simulator_mlp(env:gym.Env, sim:simulator_learn, **kwargs):
    info = sim.test_simulator(pbar=True)
    print("==========================")
    for key in info.keys():
        if key.endswith("list"):
            pass
        else:
            print(f"{key}:\t{info[key]}")
    print("==========================")
    return info

if __name__ == "__main__":
    args, env, kwargs = get_config("PRDC")
    train_model_MLP(args.env_id, env, args, kwargs)

    # sim = simulator_learn(env_id=args.env_id, env=env, model_type="MLP")
    # sim.load_model("MLP")
    # test_simulator_mlp(env, sim)
    pass
