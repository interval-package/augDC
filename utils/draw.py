import os
import matplotlib.pyplot as plt
import math
from dataclasses import dataclass
from typing import List, Tuple, Union, Dict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

"""
result:
    offline
        envs
            algs
                time
"""

path_script = os.path.abspath(__file__)
path_folder = os.path.dirname(path_script)
path_ws = os.path.join(path_folder, "..")
path_pic = os.path.join(path_ws, "pics")
if not os.path.exists(path=path_pic):
    os.mkdir(path_pic)
path_result = os.path.join(path_ws, "result")
path_offline = os.path.join(path_result, "offline")


offline_tags = \
[
    'offline/critic_loss', 
    'offline/guard_loss', 
    'offline/dc_loss', 
    'offline/cons_loss', 
    'offline/dc_cons_gap', 
    'offline/actor_loss', 
    'offline/combined_loss', 
    'offline/Q_value', 
    'offline/C_value', 
    'offline/lmbda', 
]

eval_tags = \
[
    'eval/avg_reward', 
    'eval/d4rl_score', 
    'eval/epi_len'
]

def get_child_dirs(tar_dir):
    child_folders = [name for name in os.listdir(tar_dir) if os.path.isdir(os.path.join(tar_dir, name))]
    return child_folders


def tbd_get_all_tags(tar):
    if isinstance(tar, EventAccumulator):
        event_acc = tar
    else:
        print(f"reload from f{tar}")
        event_acc = EventAccumulator(tar)
        event_acc.Reload()
    tags = event_acc.Tags()['scalars']
    print(tags)
    return tags


def tbd_get_data(event_acc, tag):
    scalar_data = event_acc.Scalars(tag)
    steps = [x.step for x in scalar_data]
    values = [x.value for x in scalar_data]
    return values, steps


@dataclass
class CurveData:
    values: list
    steps: list
    name: list


def start2plot(data:Dict[str, Dict[str, List[CurveData]]], **kwargs):
    figures = {}
    for key, val in  data.items():
        f_eval = plt.figure()
        f_eval.set_label("eval/" + key)
        f_off = plt.figure()
        f_off.set_label("offline/" + key)
        eval_idx = 1
        eval_axs = []
        off_idx = 1
        off_axs = []
        for tag, cdatas in val.items():
            if tag.startswith("eval"):
                ax = f_eval.add_subplot(1,3,eval_idx)
                for cdata in cdatas:
                    ax.plot(cdata.steps, cdata.values, label=cdata.name)
                ax.legend()
                ax.set_title(tag)
                eval_idx += 1
                eval_axs.append(ax)
            elif tag.startswith("offline"):
                ax = f_off.add_subplot(3,3,off_idx)
                for cdata in cdatas:
                    ax.plot(cdata.steps, cdata.values, label=cdata.name)
                ax.legend()
                ax.set_title(tag)
                off_idx += 1
                off_axs.append(ax)
                pass
            else:
                raise ValueError("tag invalid")
            pass
        f_off.tight_layout()
        f_eval.tight_layout()
        figures[key] = (f_off, f_eval, off_axs, eval_axs)
    return figures

def tranverse_all(envs:list = None, algs:list = None):
    # prepare data dict
    datas:Dict[str, Dict[str, List[CurveData]]] = {}

    env_folders = get_child_dirs(path_offline)
    if envs is None:
        envs = env_folders
    for env in env_folders:
        if env in envs:
            env_data:Dict[str, List[CurveData]] = {}
            algs_folders = get_child_dirs(os.path.join(path_offline, env))
            if algs is None:
                algs = algs_folders
            for alg in algs_folders:
                if alg in algs:
                    exp_times = get_child_dirs(os.path.join(path_offline, env, alg))
                    only_time = len(exp_times) == 0
                    for exp_time in exp_times:
                        acc = EventAccumulator(os.path.join(path_offline, env, alg, exp_time))
                        acc.Reload()
                        tags = tbd_get_all_tags(acc)
                        for tag in tags:
                            if tag not in env_data.keys():
                                env_data[tag] = []
                            env_data[tag].append(CurveData(*tbd_get_data(acc, tag), alg if only_time else alg + exp_times))
                        pass
                pass
            datas[env] = env_data
        pass
    return datas

def main():
    print("getting data")
    data = tranverse_all()
    print("ploting")
    figs = start2plot(data=data)
    for key, val in figs.items():
        f_off, f_eval, off_axs, eval_axs = val
        f_off.savefig(os.path.join(path_pic, f"{key}_offline.png"))
        f_eval.savefig(os.path.join(path_pic, f"{key}_eval.png"))
    return


if __name__ == "__main__":
    main()
    pass


