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
from utils.buffer import ReplayBuffer
import algs
path_script = os.path.abspath(__file__)

path_folder = os.path.dirname(path_script)

dir_out_video  = os.path.join(path_folder, "result", "video")
if not os.path.exists(dir_out_video):
    os.makedirs(dir_out_video)

def main(args, env: gym.Env, kwargs):

    # load model
    model_path = args.load_model
    load_model = True

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

    if load_model and os.path.exists(model_path):

        policy.load(model_path)
    else:
        raise ValueError("No model load.")

    # model_path = os.path.join(ckpt_dir, model_name + "_" + str(t + 1) + ".pth")
    video_path = os.path.join(dir_out_video, time.strftime("%m-%d-%H-%M") + ".gif")


    result = eval_policy(
        policy,
        args.env_id,
        args.seed,
        mean,
        std,
        save_gif=True,
        video_path=video_path,
        eval_episodes=1
    )
    return result


if __name__ == "__main__":
    args, env, kwargs = get_config_on()
    args.load_model = os.path.join("result", "buffer", args.env_id, "model.pth")
    main(args, env, kwargs)
    pass