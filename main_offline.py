import gym
import numpy as np
import time
import os
import d4rl

import algs.offline
import algs.online
from utils.eval import eval_policy
from utils.config import get_config_off, save_config
from utils.logger import get_logger, get_writer
from utils.buffer import ReplayBuffer
import algs
path_script = os.path.abspath(__file__)

path_folder = os.path.dirname(path_script)

dir_out_off = os.path.join(path_folder, "result", "offline")

def main_offline_train(args, env, kwargs):
    start_time = time.time()
    dir_result = os.path.join(
        dir_out_off,
        args.env_id,  
        args.policy,
        time.strftime("%m-%d-%H:%M:%S")
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
    next_states = replay_buffer.next_state
    actions = replay_buffer.action
    datasa = np.hstack([args.beta * states, actions])
    datas = np.vstack([states, next_states])

    more_args = {
        "data": datasa,
        "data_s": datas,
        "max_timesteps": args.max_timesteps
    }

    performed_args = more_args
    performed_args.update(kwargs)

    policy:algs.offline.AlgBaseOffline = getattr(algs.offline, args.policy)(**performed_args)

    cur_args:dict = args.__dict__
    cur_args.update(policy.alg_params)
    
    # save configs
    save_config(cur_args, os.path.join(dir_result, "config.txt"))

    evaluations = []
    evaluation_path = os.path.join(dir_result, file_name + ".npy")
    if os.path.exists(model_path):
        policy.load(model_path)

    for t in range(int(args.max_timesteps)):
        result = policy.train(replay_buffer, args.batch_size, config=cur_args)
        for key, value in result.items():
            writer.add_scalar("offline/"+key, value, global_step=t)

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
                writer.add_scalar("eval/"+key, value, global_step=t)
            logger.info("---------------------------------------")
            logger.info(f"Time steps: {t + 1}, D4RL score: {result['d4rl_score']:.2f}, Epi len: {result['epi_len']}, Rew: {result['avg_reward']:.2f}")

    np.save(evaluation_path, evaluations)
    end_time = time.time()
    logger.info(f"Total Time: {end_time - start_time}")


if __name__ == "__main__":
    args, env, kwargs = get_config_off("PRDC")
    main_offline_train(args, env, kwargs)
    pass