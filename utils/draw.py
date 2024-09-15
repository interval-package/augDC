import os
import matplotlib.pyplot as plt
import math, pickle
from dataclasses import dataclass
from typing import List, Tuple, Union, Dict
import scipy
import scipy.signal
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

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
    # 'offline/cons_loss', 
    # 'offline/dc_cons_gap', 
    'offline/actor_loss', 
    # 'offline/combined_loss', 
    'offline/Q_value', 
    'offline/C_value', 
    # 'offline/lmbda', 
]

eval_tags = \
[
    'eval/avg_reward', 
    'eval/d4rl_score', 
    'eval/epi_len'
]

def simple_filter(tag):
    tags = offline_tags + eval_tags
    return tag in tags

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
    name: str

from scipy import stats
from scipy.interpolate import make_interp_spline

def start2plot(data:Dict[str, Dict[str, List[CurveData]]], tag_filter=None, **kwargs):
    figures = {}

    def legend(fig, ax):
        handles, labels = ax.get_legend_handles_labels()

        # lower center
        fig.legend(handles, labels, loc='lower center', ncol=3)
        fig.subplots_adjust(bottom=0.15)

    def plot_cdata(cdata, ax, window=20):
        print(f"plot {key} {tag}  {cdata.name}")
        x = np.array(cdata.steps)
        y = np.array(cdata.values)
        y_smooth = scipy.signal.savgol_filter(y, window, 3)
        y_upper = [np.max(y[i:i+window]) for i in range(len(x)-window)] + [np.max(y[-window+i:]) for i in range(window)]
        y_lower = [np.min(y[i:i+window]) for i in range(len(x)-window)] + [np.min(y[-window+i:]) for i in range(window)]
        y_upper = scipy.signal.savgol_filter(np.array(y_upper), window, 3)
        y_lower = scipy.signal.savgol_filter(np.array(y_lower), window, 3)
        line = ax.plot(x, y_smooth, label=cdata.name, alpha=0.8)[0]
        color = line.get_color()
        ax.fill_between(x, y_lower, y_upper, color=color, alpha=0.2)

    for key, val in  data.items():
        f_eval = plt.figure(figsize=(16, 9))
        f_eval.set_label("eval/" + key)
        f_off = plt.figure(figsize=(16, 9))
        f_off.set_label("offline/" + key)
        eval_idx = 1
        eval_axs = []
        off_idx = 1
        off_axs = []
        for tag, cdatas in val.items():
            if tag_filter is not None and not tag_filter(tag):
                print(f"{tag} is filtered")
                continue
            if tag.startswith("eval"):
                ax = f_eval.add_subplot(3,1,eval_idx)
                for cdata in cdatas:
                    if cdata.name.startswith("PRWIC_sum"):
                        continue
                    plot_cdata(cdata, ax, 20)
                # ax.legend()
                ax.set_title(tag)
                eval_idx += 1
                eval_axs.append(ax)
            elif tag.startswith("offline"):
                ax = f_off.add_subplot(2,3,off_idx)
                for cdata in cdatas:
                    if cdata.name.startswith("PRWIC_sum"):
                        continue
                    plot_cdata(cdata, ax, 200)
                ax.set_title(tag)
                off_idx += 1
                off_axs.append(ax)
                pass
            else:
                raise ValueError("tag invalid")
            pass
        f_off.tight_layout()
        f_eval.tight_layout()
        legend(f_off, off_axs[0])
        legend(f_eval, eval_axs[0])
        figures[key] = (f_off, f_eval, off_axs, eval_axs)
    return figures

def tranverse_all(envs:list = None, algs:list = None, cached=False):
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
                    only_time = len(exp_times) == 1
                    for exp_time in exp_times:
                        print(f"Processing {env} {alg} {exp_time}")
                        if cached and os.path.exists(os.path.join(path_offline, env, alg, exp_time, "buffered.pkl")):
                            print("Loading...")
                            with open(os.path.join(path_offline, env, alg, exp_time, "buffered.pkl"), "rb") as f:
                                cdata_dict = pickle.load(f)
                            for ckey, cval in cdata_dict.items():
                                if ckey not in env_data.keys():
                                    env_data[ckey] = []
                                env_data[ckey] = env_data[ckey] + cval
                        else:
                            acc = EventAccumulator(os.path.join(path_offline, env, alg, exp_time))
                            acc.Reload()
                            tags = tbd_get_all_tags(acc)
                            temp_dict = {}
                            for tag in tags:
                                if tag not in env_data.keys():
                                    env_data[tag] = []
                                temp_dict[tag] = []
                                cdata = CurveData(*tbd_get_data(acc, tag), alg if only_time else alg + exp_time)
                                temp_dict[tag].append(cdata)
                                env_data[tag].append(cdata)
                            with open(os.path.join(path_offline, env, alg, exp_time, "buffered.pkl"), "wb") as f:
                                pickle.dump(temp_dict, f)
                        pass
                pass
            datas[env] = env_data
        pass
    return datas

def main():
    sns.set_style("whitegrid")
    print("getting data")
    data = tranverse_all(cached=True)
    print("ploting")
    figs = start2plot(data=data, tag_filter=simple_filter)
    for key, val in figs.items():
        f_off, f_eval, off_axs, eval_axs = val
        f_off.savefig(os.path.join(path_pic, f"{key}_offline.png"))
        f_eval.savefig(os.path.join(path_pic, f"{key}_eval.png"))
    return


if __name__ == "__main__":
    main()
    pass


