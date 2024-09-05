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
    "halfcheetah-medium-expert-v2",
    "halfcheetah-medium-replay-v2",
    "halfcheetah-medium-v2",
    "halfcheetah-random-v2",
    "hopper-medium-expert-v2",
    "hopper-medium-replay-v2",
    "hopper-medium-v2",
    "hopper-random-v2",
    "maze2d-umaze-sparse-v1",
    "walker2d-medium-expert-v2",
    "walker2d-medium-replay-v2",
    "walker2d-medium-v2",
    "walker2d-random-v2",
]

scripts = [
    "main_offline.py",
    # "main_online.py",
]

policies = [
    # "PRDC",
    "PRWIC"
]

path_script = os.path.abspath(__file__)

path_folder = os.path.dirname(path_script)

path_tasks = os.path.join(path_folder, "tasks.json")
path_config = os.path.join(path_folder, "configs")

def generate_task_config(policy, env_id, **kwargs)->dict:
    ret = {
        "policy": policy,
        "env_id": env_id,
    }
    ret.update(kwargs)
    return ret

if __name__ == "__main__":
    tasks = []
    general_config = {
        "device": "cpu"
    }
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