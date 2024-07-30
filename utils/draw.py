import os
import matplotlib.pyplot as plt
import math
from dataclasses import dataclass
from typing import Union, Dict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

path_script = os.path.abspath(__file__)

path_folder = os.path.dirname(path_script)

path_folder = "/home/zhangfeihong/code/zza/yolov5/runs/train/coco"

path_pic = os.path.join(path_folder, "pics")
if not os.path.exists(path=path_pic):
    os.mkdir(path_pic)

child_folders = [name for name in os.listdir(path_folder) if os.path.isdir(os.path.join(path_folder, name))]

tags_train = ['train/box_loss', 'train/obj_loss', 'train/cls_loss']
tags_val = ['val/box_loss', 'val/obj_loss', 'val/cls_loss']
tags_metric = ['metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95']
tags_lr = ['x/lr0', 'x/lr1', 'x/lr2']

tags_dict = {
    "tags_TrainLoss": tags_train, 
    "tags_ValLoss": tags_val, 
    "tags_metric": tags_metric, 
    "tags_lr": tags_lr
    }

def extract_data_tag(log_dir, tag):
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    print(event_acc.Tags()['scalars'])
    if tag not in event_acc.Tags()['scalars']:
        raise ValueError(f"Tag {tag} not found in TensorBoard logs.")

    scalar_data = event_acc.Scalars(tag)
    steps = [x.step for x in scalar_data]
    values = [x.value for x in scalar_data]

    return steps, values

@dataclass
class UniformedTbdData:
    info: Dict[str, Dict[str, list]]
    steps: list
    log_dir: str

    @property
    def name(self):
        return os.path.basename(self.log_dir)

def extract_data_uniformed(log_dir, **kwargs)->UniformedTbdData:
    """
    return: 
    {
        "info":{
            "tags 1": {"tag1": ...},
            ...,
        }
        "steps": list
        "log_dir": log_dir
    }
    """
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    info = {}
    steps = None
    for key, val in tags_dict.items():
        temp = {}
        for tag in val:
            scalar_data = event_acc.Scalars(tag)
            if steps is None:
                steps = [x.step for x in scalar_data]
            values = [x.value for x in scalar_data]
            temp[tag] = values
        info[key] = temp
    assert steps is not None, "Steps not defined."
    return UniformedTbdData(info, steps, log_dir)

def get_dirs():
    ret = []
    for folder in os.listdir(path_folder):
        if os.path.isdir(os.path.join(path_folder, folder)):
            ret.append(folder)
    return ret

@dataclass
class PlotFigure:
    fig: plt.Figure
    axs: Dict[str, plt.Axes]
    key: str

    truncate:int=None
    alpha=0.6
    axe_legend=None
    is_legend=True

    def set_plot_config(self, **kwargs):
        if "truncate" in kwargs:
            self.truncate = kwargs["truncate"]
        if "alpha" in kwargs:
            self.alpha = kwargs["alpha"]

    @staticmethod
    def prepare_axes(fig: plt.Figure, tags:list):
        edge_len = math.ceil(math.sqrt(len(tags)))
        ret = {}
        for i, tag in enumerate(tags):
            ax:plt.Axes = fig.add_subplot(edge_len, edge_len, i+1)
            ret[tag] = ax
            ax_name = tag.split("/")[-1]
            ax.set_title(ax_name)
        return ret

    def plotTbd(self, data:UniformedTbdData):
        steps = data.steps
        name = data.name
        last_idx = -1 if self.truncate is None else min(self.truncate, len(steps))
        values_dict = data.info[self.key]
        for key, val in values_dict.items():
            ax:plt.Axes = self.axs[key]
            ax.plot(steps[:last_idx], val[:last_idx], label=name, alpha=self.alpha)
            pass
        return

    def tidy(self):
        for key, val in self.axs.items():
            val.grid()
        if self.is_legend:
            if self.axe_legend is None:
                l = math.sqrt(len(self.axs))
                if math.ceil(l) > l:
                    l = math.ceil(l)
                    self.axe_legend = self.fig.add_subplot(l,l,l**2)
                else:
                    self.axe_legend = val
            handles, labels = val.get_legend_handles_labels()
            self.axe_legend.legend(handles, labels, loc='best')
        self.fig.tight_layout()

    def save_fig(self):
        fig = self.fig
        fig.savefig(os.path.join(path_pic, f"{self.key}.png"))

    pass

def plot_all(dirs:list, **kwargs):
    
    # prepare fig dict
    fig_dict:Dict[str, PlotFigure] = {}
    for key, val in tags_dict.items():
        title = key.split("_")[-1]
        fig = plt.figure()
        fig.suptitle(title)
        pfig = PlotFigure(fig, PlotFigure.prepare_axes(fig, val), key)
        pfig.set_plot_config(**kwargs)
        if key == "tags_metric":
            pfig.is_legend=False
        fig_dict[key] = pfig

    for dir in dirs:
        data = extract_data_uniformed(dir)
        for key in data.info.keys():
            pfig = fig_dict[key]
            pfig.plotTbd(data)

    for key, val in fig_dict.items():
        val.tidy()
        val.save_fig()
    return


if __name__ == "__main__":
    # ret = extract_data_uniformed(os.path.join(path_folder, "AdamRad_cmp_1"))
    dirs = [
        "AdamW_lre3", 
        "AdamWRad_lre3", 
        "Adam_lre2", 
        "Adam_lre3", 
        "Rad_lre3",
        "Rad_lre2",
        "Rad_lre53",
        ]

    # dirs = [
    #     "Rad_lre2_zetae16",
    #     "Rad_lre2_nzeta",
    #     "Rad_nparam",
    #     "Rad_miter3000"
    # ]

    # dirs = [
    #     "Rad_lre2_zetae16",
    #     "AdamTorch_lre2",
    #     "AdamWTorch_lre2",
    #     "SGDTorch_lre2"
    # ]

    dirs = [os.path.join(path_folder, dir) for dir in dirs]
    plot_all(dirs, truncate=300)

