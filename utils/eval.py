import gym
import d4rl
import numpy as np
import imageio

fps = 30

def eval_policy(
    policy,
    env_name,
    seed,
    mean=0,
    std=1,
    seed_offset=100,
    eval_episodes=50,
    save_gif=False,
    video_path=None,
):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + seed_offset)

    avg_reward = 0.0
    ep_obs = []
    epi_len = 0
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        step = 0
        while not done:
            if save_gif and _ == 0:
                obs = eval_env.render(mode="rgb_array")
                ep_obs.append(obs)
            state = (np.array(state).reshape(1, -1) - mean) / std
            action = policy.select_action(state)
            state, reward, done, __ = eval_env.step(action)
            avg_reward += reward
            step += 1
        epi_len += step

    avg_reward /= eval_episodes
    epi_len /= eval_episodes
    d4rl_score = eval_env.get_normalized_score(avg_reward) * 100
    if save_gif:
        with imageio.get_writer(video_path, fps=fps) as writer:
            for obs in ep_obs:
                writer.append_data(obs)

    info = {
        "avg_reward": avg_reward,
        "d4rl_score": d4rl_score,
        "epi_len": epi_len
    }
    return info
