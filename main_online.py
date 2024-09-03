import gym
import numpy as np
import time
import os
import d4rl

import algs.offline
import algs.online
from utils.eval import eval_policy
from utils.config import get_config_on, save_config
from utils.logger import get_logger, get_writer
from utils.buffer import ReplayBuffer, OnlineSampler
import algs
path_script = os.path.abspath(__file__)

path_folder = os.path.dirname(path_script)

dir_out_on  = os.path.join(path_folder, "result", "online")

"""
Trainning with online finetuning not only straightly train with online data, 
it also could be a combine of each
"""

def main_online_fintune(args, env: gym.Env, kwargs):
    dir_result = os.path.join(
        dir_out_on,
        args.env_id,
        time.strftime("%m-%d-%H:%M:%S")
        + "_"
        + args.policy
        + "_"
        + str(args.seed),
    )

    if not os.path.exists(dir_result):
        os.makedirs(dir_result, exist_ok=True)

    writer = get_writer(dir_result)

    file_name = f"{args.policy}_{args.env_id}_{args.seed}"
    logger = get_logger(os.path.join(dir_result, file_name + ".log"))
    logger.info(
        f"Policy: {args.policy}, Env: {args.env_id}, Seed: {args.seed}, Info: {args.info}"
    )

    # load model
    ckpt_dir = os.path.join(dir_result, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    load_model = False
    model_name = file_name
    if args.load_model != "default":
        model_path = args.load_model
        load_model = True
    else:
        model_path = os.path.join(ckpt_dir, model_name + ".pth")
    
    # save configs
    args.load_model = load_model
    save_config(args, os.path.join(dir_result, "config.txt"))

    rep_config = {
        "state_dim": kwargs["state_dim"], 
        "action_dim": kwargs["action_dim"], 
        "device": args.device, 
        "env_id": args.env_id, 
        "scale": args.scale, 
        "shift": args.shift
    }

    replay_buffer = ReplayBuffer(**rep_config)
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
    if args.normalize:
        mean, std = replay_buffer.normalize_states()
    else:
        mean, std = 0, 1

    sampler = OnlineSampler(**rep_config)

    states = replay_buffer.state
    actions = replay_buffer.action
    data = np.hstack([args.beta * states, actions])
    kwargs["data"] = data
    policy:algs.online.AlgBaseOnline = getattr(algs.online, args.policy)(env=env, **kwargs)

    evaluations = []
    evaluation_path = os.path.join(dir_result, file_name + ".npy")
    if load_model and os.path.exists(model_path):
        logger.info(f"Loading model at {model_path}.")
        policy.load(model_path)
    else:
        logger.info("Not loading model.")

    for t in range(int(args.max_episode)):
        result = policy.train(step=t, max_epi_len=args.max_epi_len)
        for key, value in result.items():
            writer.add_scalar(key, value, global_step=t)

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            model_path = os.path.join(ckpt_dir, model_name + "_" + str(t + 1) + ".pth")
            video_path = os.path.join(ckpt_dir, model_name + "_" + str(t + 1) + ".gif")

            if args.save_model and (t + 1) % args.save_model_freq == 0:
                result = eval_policy(
                    policy,
                    args.env_id,
                    args.seed,
                    mean,
                    std,
                    save_gif=False,
                    video_path=video_path,
                )
                policy.save(model_path)
            else:
                result = eval_policy(
                    policy, args.env_id, args.seed, mean, std
                )
            for key, value in result.items():
                writer.add_scalar("eval/" + key, value, global_step=t)
            logger.info("---------------------------------------")
            logger.info(f"Time steps: {t + 1}, D4RL score: {result['d4rl_score']}, Epi len: {result['epi_len']}, Rew: {result['avg_reward']}")

    pass

if __name__ == "__main__":
    args, env, kwargs = get_config_on()
    args.load_model = os.path.join("result", "buffer", args.env_id, "model.pth")
    main_online_fintune(args, env, kwargs)
    pass