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

def simple_filter_tag(tag):
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

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def fill_between_3d(ax, x, y, z1, z2, color="orange", alpha=0.2):
    # 创建填充区域
    verts = [list(zip(x, y, z1)), list(zip(x, y, z2))]  # 顶点
    verts_combined = list(zip(x, y, z1)) + list(zip(x[::-1], y[::-1], z2[::-1]))  # 组合顶点

    # 填充区域 (使用Poly3DCollection)
    poly = Poly3DCollection([verts_combined], color=color, alpha=alpha)
    ax.add_collection3d(poly)

def plot_cdata(cdata, ax, window=20, plot3d=False):
    x = np.array(cdata.steps)
    y = np.array(cdata.values)
    y_smooth = scipy.signal.savgol_filter(y, window, 3)
    y_upper = [np.max(y[i:i+window]) for i in range(len(x)-window)] + [np.max(y[-window+i:]) for i in range(window)]
    y_lower = [np.min(y[i:i+window]) for i in range(len(x)-window)] + [np.min(y[-window+i:]) for i in range(window)]
    y_upper = scipy.signal.savgol_filter(np.array(y_upper), window, 3)
    y_lower = scipy.signal.savgol_filter(np.array(y_lower), window, 3)
    if plot3d:
        z = np.ones_like(x) * ax._user_z_idx 
        line = ax.plot(z, x, y_smooth, label=cdata.name, alpha=0.8)[0]
        color = line.get_color()
        fill_between_3d(ax, z, x, y_lower, y_upper, color=color, alpha=0.2)
        ax._user_z_idx += 1
    else:
        line = ax.plot(x, y_smooth, label=cdata.name, alpha=0.8)[0]
        color = line.get_color()
        ax.fill_between(x, y_lower, y_upper, color=color, alpha=0.2)
    return x, y_smooth, y_lower, y_upper

def legend_fig(fig, ax):
    handles, labels = ax.get_legend_handles_labels()
    # lower center
    fig.legend(handles, labels, loc='lower center', ncol=3)
    fig.subplots_adjust(bottom=0.15)

from matplotlib.figure import Figure

@dataclass
class axe_info:
    fig: Figure
    cast: Dict[str, plt.Axes]
    idx: int

    def append(self, tag:str, arr:Tuple[int,int], **kwargs):
        if tag not in self.cast.keys():
            self.cast[tag] = self.fig.add_subplot(*arr, self.idx, **kwargs)
            self.idx += 1
            self.cast[tag]._user_z_idx = 0
        return self.cast[tag]

    def get_legend_handles_labels(self):
        max_h, max_l = [], []
        for key, val in self.cast.items():
            handles, labels = val.get_legend_handles_labels()
            if len(labels) > len(max_l):
                max_h, max_l = handles, labels
        return max_h, max_l

def start2plot_2d(data:Dict[str, Dict[str, List[CurveData]]], **kwargs):
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

def start2plot_2d_all(data:Dict[str, Dict[str, List[CurveData]]], tag_filter=None, **kwargs):
    figures = {}

    for key, val in  data.items():
        env_type = env_type = key.split("-")[0]
        if env_type in figures.keys():
            f_off, f_eval, off_axs, eval_axs = figures[env_type]
        else:
            f_eval = plt.figure(figsize=(16, 9))
            f_eval.set_label("eval/" + key)
            f_off = plt.figure(figsize=(16, 9))
            f_off.set_label("offline/" + key)
            eval_axs = axe_info(f_eval, {}, 1)
            off_axs = axe_info(f_off, {}, 1)
            figures[env_type] = (f_off, f_eval, off_axs, eval_axs)
        for tag, cdatas in val.items():
            if tag_filter is not None and not tag_filter(tag):
                print(f"{tag} is filtered")
                continue
            if tag.startswith("eval"):
                ax = eval_axs.append(tag, (3, 1))
                for cdata in cdatas:
                    if cdata.name.startswith("PRWIC_sum"):
                        continue
                    plot_cdata(cdata, ax, 20)
                ax.set_title(tag)
            elif tag.startswith("offline"):
                ax = off_axs.append(tag, (2, 3))
                for cdata in cdatas:
                    if cdata.name.startswith("PRWIC_sum"):
                        continue
                    plot_cdata(cdata, ax, 200)
                ax.set_title(tag)
                pass
            else:
                raise ValueError("tag invalid")
            pass
        f_off.tight_layout()
        f_eval.tight_layout()

    for key, val in figures.items():
        f_off, f_eval, off_axs, eval_axs = val
        legend_fig(f_off, off_axs)
        legend_fig(f_eval, eval_axs)
        
    return figures

def start2plot_3d(data:Dict[str, Dict[str, List[CurveData]]], tag_filter=None, **kwargs):
    figures = {}

    for key, val in  data.items():
        env_type = env_type = key.split("-")[0]
        if env_type in figures.keys():
            f_off, f_eval, off_axs, eval_axs = figures[env_type]
        else:
            f_eval = plt.figure(figsize=(16, 9))
            f_eval.set_label("eval/" + key)
            f_off = plt.figure(figsize=(16, 9))
            f_off.set_label("offline/" + key)
            eval_axs = axe_info(f_eval, {}, 1)
            off_axs = axe_info(f_off, {}, 1)
            figures[env_type] = (f_off, f_eval, off_axs, eval_axs)
        for tag, cdatas in val.items():
            if tag_filter is not None and not tag_filter(tag):
                print(f"{tag} is filtered")
                continue
            if tag.startswith("eval"):
                ax = eval_axs.append(tag, (1, 3), projection='3d')
                for cdata in cdatas:
                    if cdata.name.startswith("PRWIC_sum"):
                        continue
                    cdata.name = key + "_" + cdata.name
                    plot_cdata(cdata, ax, 20, True)
                ax.set_title(tag)
            elif tag.startswith("offline"):
                ax = off_axs.append(tag, (2, 3), projection='3d')
                for cdata in cdatas:
                    if cdata.name.startswith("PRWIC_sum"):
                        continue
                    cdata.name = key + "_" + cdata.name
                    plot_cdata(cdata, ax, 200, True)
                ax.set_title(tag)
                pass
            else:
                raise ValueError("tag invalid")
            pass
        f_off.tight_layout()
        f_eval.tight_layout()

    for key, val in figures.items():
        f_off, f_eval, off_axs, eval_axs = val
        legend_fig(f_off, off_axs)
        legend_fig(f_eval, eval_axs)
        
    return figures

def tranverse_all(path_data=path_offline, envs:list = None, algs:list = None, cached=False):
    # prepare data dict
    datas:Dict[str, Dict[str, List[CurveData]]] = {}

    env_folders = get_child_dirs(path_data)
    if envs is None:
        envs = env_folders
    for env in env_folders:
        if env in envs:
            env_data:Dict[str, List[CurveData]] = {}
            algs_folders = get_child_dirs(os.path.join(path_data, env))
            if algs is None:
                algs = algs_folders
            for alg in algs_folders:
                if alg in algs:
                    exp_times = get_child_dirs(os.path.join(path_data, env, alg))
                    only_time = len(exp_times) == 1
                    for exp_time in exp_times:
                        print(f"Processing {env} {alg} {exp_time}")
                        if cached and os.path.exists(os.path.join(path_data, env, alg, exp_time, "buffered.pkl")):
                            print("Loading...")
                            with open(os.path.join(path_data, env, alg, exp_time, "buffered.pkl"), "rb") as f:
                                cdata_dict = pickle.load(f)
                            for ckey, cval in cdata_dict.items():
                                if ckey not in env_data.keys():
                                    env_data[ckey] = []
                                env_data[ckey] = env_data[ckey] + cval
                        else:
                            acc = EventAccumulator(os.path.join(path_data, env, alg, exp_time))
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
                            with open(os.path.join(path_data, env, alg, exp_time, "buffered.pkl"), "wb") as f:
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
    figs = start2plot_3d(data=data, tag_filter=simple_filter_tag)
    for key, val in figs.items():
        f_off, f_eval, off_axs, eval_axs = val
        f_off.savefig(os.path.join(path_pic, f"{key}_offline.png"))
        f_eval.savefig(os.path.join(path_pic, f"{key}_eval.png"))
    return


def save_tranverse(tar_path=path_offline, res_path="./res.pkl"):
    data = tranverse_all(cached=True)
    with open("./res.pkl", "wb") as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    main()
    pass


