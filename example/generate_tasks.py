import argparse
import os, json

datasets = \
[
    # "Ant-maze-big-maze-noisy-multistart-True-multigoal-False-sparse-fixed",
    # "Ant-maze-big-maze-noisy-multistart-True-multigoal-False-sparse",
    # "Ant-maze-big-maze-noisy-multistart-True-multigoal-True-sparse-fixed",
    # "Ant-maze-big-maze-noisy-multistart-True-multigoal-True-sparse",
    # "Ant-maze-hardest-maze-noisy-multistart-True-multigoal-False-sparse",
    # "Ant-maze-hardest-maze-noisy-multistart-True-multigoal-True-sparse",
    # "Ant-maze-u-maze-noisy-multistart-False-multigoal-False-sparse-fixed",
    # "Ant-maze-u-maze-noisy-multistart-False-multigoal-False-sparse",
    # "Ant-maze-u-maze-noisy-multistart-True-multigoal-True-sparse-fixed",
    # "Ant-maze-u-maze-noisy-multistart-True-multigoal-True-sparse",
    # "halfcheetah-medium-expert-v2",
    # "halfcheetah-medium-replay-v2",
    # "halfcheetah-medium-v2",
    # "halfcheetah-random-v2",
    "hopper-medium-expert-v2",
    "hopper-medium-replay-v2",
    "hopper-medium-v2",
    "hopper-random-v2",
    # "maze2d-umaze-sparse-v1",
    # "walker2d-medium-expert-v2",
    # "walker2d-medium-replay-v2",
    # "walker2d-medium-v2",
    # "walker2d-random-v2",
]

param_env = {
    "hopper": {
        "beta": 2.0,
        "alpha": 2.5,
    },
    "halfcheetah": {
        "alpha": 40.0,
        "beta": 2.0,
        "k": 1,
    },
    "walker2d": {},
    "antmaze": {},
}

scripts = [
    "main_offline.py",
    # "main_online.py",
]

policies = [
    # "PRDC",
    # "PRWIC_sum",
    "PRWIC_max"
]

path_script = os.path.abspath(__file__)

path_folder = os.path.dirname(path_script)

path_tasks = os.path.join(path_folder, "tasks.json")
path_config = os.path.join(path_folder, "configs")

path_config_datasets = os.path.join(path_folder, "datasets.json")
path_config_scripts = os.path.join(path_folder, "scripts.json")
path_config_policies = os.path.join(path_folder, "policies.json")

def generate_task_config(policy, env_id, **kwargs)->dict:
    ret = {
        "policy": policy,
        "env_id": env_id,
    }
    env_name = env_id.split("-")[0]
    if env_name in param_env.keys():
        ret.update(param_env[env_name])
    ret.update(kwargs)
    return ret

if __name__ == "__main__":
    tasks = []
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", default=None)
    args = parser.parse_args()
    args = vars(args)
    general_config = {}
    for key, val in args.items():
        if val is not None:
            general_config[key] = val

    # find have json config or not
    if os.path.exists(path_config_datasets):
        print(f"Loading from {path_config_datasets}...")
        with open(path_config_datasets, "rt") as f:
            datasets = json.load(f)["data"]
    if os.path.exists(path_config_scripts):
        print(f"Loading from {path_config_scripts}...")
        with open(path_config_scripts, "rt") as f:
            scripts = json.load(f)["data"]
    if os.path.exists(path_config_policies):
        print(f"Loading from {path_config_policies}...")
        with open(path_config_policies, "rt") as f:
            policies = json.load(f)["data"]
    
    for script in scripts:
        for policy in policies:
            for env_id in datasets:
                config = generate_task_config(policy, env_id, **general_config)
                tasks.append(
                    {
                        "script": script,
                        "config": config
                    }
                )

    with open(path_tasks, "wt") as taskf:
        json.dump({
            "info": "tasks",
            "tasks":tasks
            }, taskf)
        pass
    pass