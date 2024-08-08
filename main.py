import gym
import numpy as np
import time
import os
import d4rl

import algs.offline
import algs.online
from utils.eval import eval_policy
from utils.config import get_config_off, get_config_on, save_config
from utils.logger import get_logger, get_writer
from utils.buffer import ReplayBuffer
import algs
path_script = os.path.abspath(__file__)

path_folder = os.path.dirname(path_script)

dir_out_off = os.path.join(path_folder, "result", "offline")
dir_out_on = os.path.join(path_folder, "result", "online")
if not os.path.exists(dir_out_off):
    os.makedirs(dir_out_off, exist_ok=True)
if not os.path.exists(dir_out_on):
    os.makedirs(dir_out_on, exist_ok=True)

def main_offline_train(args, env, kwargs):
    start_time = time.time()
    dir_result = os.path.join(
        dir_out_off,
        time.strftime("%m-%d-%H:%M:%S")
        + "_"
        + args.policy
        + "_"
        + args.env_id
        + "_"
        + str(args.seed),
    )

    writer = get_writer(dir_result)

    file_name = f"{args.policy}_{args.env_id}_{args.seed}"
    logger = get_logger(os.path.join(dir_result, file_name + ".log"))
    logger.info(
        f"Policy: {args.policy}, Env: {args.env_id}, Seed: {args.seed}, Info: {args.info}"
    )

    # save configs
    save_config(args, os.path.join(dir_result, "config.txt"))

    # load model
    if args.load_model != "default":
        model_name = args.load_model
    else:
        model_name = file_name
    ckpt_dir = os.path.join(dir_result, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    model_path = os.path.join(ckpt_dir, model_name + ".pth")
    
    replay_buffer = ReplayBuffer(kwargs["state_dim"], kwargs["action_dim"], args.device, args.env_id, args.scale, args.shift)
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
    if args.normalize:
        mean, std = replay_buffer.normalize_states()
    else:
        mean, std = 0, 1

    states = replay_buffer.state
    actions = replay_buffer.action
    data = np.hstack([args.beta * states, actions])

    policy:algs.offline.AlgBaseOffline = getattr(algs.offline, args.policy)(data, **kwargs)

    evaluations = []
    evaluation_path = os.path.join(dir_result, file_name + ".npy")
    if os.path.exists(model_path):
        policy.load(model_path)

    for t in range(int(args.max_timesteps)):
        result = policy.train(replay_buffer, args.batch_size)
        for key, value in result.items():
            writer.add_scalar(key, value, global_step=t)

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            model_path = os.path.join(ckpt_dir, model_name + "_" + str(t + 1) + ".pth")
            video_path = os.path.join(ckpt_dir, model_name + "_" + str(t + 1) + ".gif")

            if args.save_model and (t + 1) % args.save_model_freq == 0:
                avg_reward, d4rl_score = eval_policy(
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
                avg_reward, d4rl_score = eval_policy(
                    policy, args.env_id, args.seed, mean, std
                )
            writer.add_scalar("avg_reward", avg_reward, global_step=t)
            writer.add_scalar("d4rl_score", d4rl_score, global_step=t)
            evaluations.append(d4rl_score)
            logger.info("---------------------------------------")
            logger.info(f"Time steps: {t + 1}, D4RL score: {d4rl_score}")

    np.save(evaluation_path, evaluations)
    end_time = time.time()
    logger.info(f"Total Time: {end_time - start_time}")

def main_online_fintune(args, env: gym.Env, kwargs):
    dir_result = os.path.join(
        dir_out_on,
        time.strftime("%m-%d-%H:%M:%S")
        + "_"
        + args.policy
        + "_"
        + args.env_id
        + "_"
        + str(args.seed),
    )
    writer = get_writer(dir_result)

    file_name = f"{args.policy}_{args.env_id}_{args.seed}"
    logger = get_logger(os.path.join(dir_result, file_name + ".log"))
    logger.info(
        f"Policy: {args.policy}, Env: {args.env_id}, Seed: {args.seed}, Info: {args.info}"
    )

    # save configs
    save_config(args, os.path.join(dir_result, "config.txt"))

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
    
    replay_buffer = ReplayBuffer(kwargs["state_dim"], kwargs["action_dim"], args.device, args.env_id, args.scale, args.shift)
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
    if args.normalize:
        mean, std = replay_buffer.normalize_states()
    else:
        mean, std = 0, 1

    states = replay_buffer.state
    actions = replay_buffer.action
    data = np.hstack([args.beta * states, actions])
    kwargs["data"] = data
    policy:algs.online.AlgBaseOnline = getattr(algs.online, args.policy)(env=env, **kwargs)

    evaluations = []
    evaluation_path = os.path.join(dir_result, file_name + ".npy")
    if os.path.exists(model_path):
        logger.info("Not load model.")
        policy.load(model_path)

    for t in range(int(args.max_episode)):
        result = policy.train(env)
        for key, value in result.items():
            writer.add_scalar(key, value, global_step=t)

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            model_path = os.path.join(ckpt_dir, model_name + "_" + str(t + 1) + ".pth")
            video_path = os.path.join(ckpt_dir, model_name + "_" + str(t + 1) + ".gif")

            if args.save_model and (t + 1) % args.save_model_freq == 0:
                avg_reward, d4rl_score = eval_policy(
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
                avg_reward, d4rl_score = eval_policy(
                    policy, args.env_id, args.seed, mean, std
                )
            writer.add_scalar("avg_reward", avg_reward, global_step=t)
            writer.add_scalar("d4rl_score", d4rl_score, global_step=t)
            evaluations.append(d4rl_score)
            logger.info("---------------------------------------")
            logger.info(f"Time steps: {t + 1}, D4RL score: {d4rl_score}")

    pass

if __name__ == "__main__":
    # args, env, kwargs = get_config_off()
    # main_offline_train(args, env, kwargs)
    args, env, kwargs = get_config_on()
    args.load_model = "/root/code/augDC/result/offline/08-07-16:09:26_PRDC_halfcheetah-medium-v2_1024/ckpt/PRDC_halfcheetah-medium-v2_1024_50000.pth"
    main_online_fintune(args, env, kwargs)
    pass
