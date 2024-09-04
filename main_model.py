import os
import gym
import torch
import json
import pickle
import d4rl

from utils.buffer import ReplayBuffer
from utils.config import get_config_off, save_config
from utils.logger import add_scalars

from torch.utils.tensorboard import SummaryWriter

from simulator import simulator_learn
from world_model.model import model_MLP
from simulator.train import train_model, test_simulator

"""
This script is used to train and test a simulator.
"""

def exp_train(model_type="MLP"):
    args, env, kwargs = get_config_off("PRDC")
    mdl, config, sim = train_model(args.env_id, env, args, kwargs, model_type=model_type)
    return mdl, config, sim

def exp_test(model_type="MLP", sim: simulator_learn=None):
    args, env, kwargs = get_config_off("PRDC")
    sim.load_model(model_type)
    test_simulator(env, sim)


if __name__ == "__main__":

    model_type = "MLP"
    _, _, sim = exp_train(model_type)
    # exp_test("MLP")
    simulator_learn
    exp_test(model_type, sim=sim)

    pass
