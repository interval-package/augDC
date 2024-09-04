import json
import os
import gym
import torch
from utils.buffer import ReplayBuffer
from utils.logger import add_scalars

from torch.utils.tensorboard import SummaryWriter

import world_model.model as models
from simulator.simulator_learn import simulator_base, simulator_learn

def save_model_config(model_config:dict, path_model_config, name, save=True):
    ret = {}
    for key, value in model_config.items():
        if callable(value):
            v = value.__name__
        else:
            v = value
        ret[key] = v
        pass
    if save:
        with open(os.path.join(path_model_config, name), "wt") as f:
            json.dump(ret, f)
    return ret

def train_model(env_id, env, obj_config, dict_config, model_type="MMLP", save_model=True, **kwargs):
    device = torch.device(dict_config["device"])

    state_dim  = dict_config["state_dim"]
    action_dim = dict_config["action_dim"]
    # max_action = kwargs["max_action"]

    sim = simulator_learn(env_id=obj_config.env_id, env=env, model_type=model_type)
    sim.make_path()
    writer = SummaryWriter(log_dir=sim.path, flush_secs=120)

    def eval_func(mdl:models.model_base, iter:int, **kwargs):
        sim.set_model(mdl)
        info = sim.test_simulator(info={"iter":iter})
        if "loss" in kwargs.keys():
            info["loss"] = kwargs["loss"]
        add_scalars(info, writer=writer, step=iter)
        return info
    
    def save_func(mdl:models.model_base, iter:int, **kwargs):
        if save_model:
            sim.set_model(mdl)
            sim.save(iter)
        return {"msg": f"Saved f{iter}." if save_model else "Not save."}

    model_config = {
        "mlp_shape": [state_dim + action_dim, 256, 256, state_dim + 2],
        "lr": 0.01,
        "batch_size": 256,
        "epoches": 10000,
        "eval_round": 1000,
        "save_round": 1000,
        "model_type": model_type,
        "eval_func": eval_func,
        "save_func": save_func
    }

    save_model_config(model_config, sim.path, "model.json")
    save_model_config(vars(obj_config), sim.path, "env.json")

    model:models.model_base = getattr(models, f"model_{model_type}")(device, **model_config)

    replay_buffer = ReplayBuffer(dict_config["state_dim"], dict_config["action_dim"], device, env_id, obj_config.scale, obj_config.shift)
    replay_buffer.from_d4rlenv(env)

    model.train(replay_buffer, **model_config)
    # model.save(path_model)

    return model, model_config, sim

def test_simulator(env:gym.Env, sim:simulator_learn, **kwargs):
    info = sim.test_simulator(pbar=True)
    print("==========================")
    for key in info.keys():
        if key.endswith("list"):
            pass
        else:
            print(f"{key}:\t{info[key]}")
    print("==========================")
    return info