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

from simulator import simulator_learn
from simulator.model import model_MLP
from simulator.train import train_model, test_simulator

"""
This script is used to train and test a simulator.
"""

def exp_train():
    args, env, kwargs = get_config("PRDC")
    train_model(args.env_id, env, args, kwargs, model_type="MLP")
    train_model(args.env_id, env, args, kwargs, model_type="MMLP")

def exp_test(model_type="MLP"):
    args, env, kwargs = get_config("PRDC")
    sim = simulator_learn(env_id=args.env_id, env=env, model_type=model_type)
    sim.load_model(model_type)
    test_simulator(env, sim)


if __name__ == "__main__":

    # exp_train()
    exp_test("MLP")
    exp_test("MMLP")

    pass
