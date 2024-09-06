import os
import json
import torch
import world_model.model as models
from world_model.model import model_base, model_MMLP
from utils.np2t import Transition

"""
This file model is different from the simulator part.
Allow grad backward.
"""

class WorldModel:
    """
    # WorldModel
    WorldModel do not include training scripts, training using the `simulator_learn`.
    Do not set as nn.module but holds that part.
    """
    def __init__(self, model_config, env_config, **kwargs) -> None:
        pass

    model: model_base

    @classmethod
    def load(cls, mpath, iter, device, **kwargs):
        with open(os.path.join(mpath, "env.json"), "rt") as f:
            env_config = json.load(f)
        with open(os.path.join(mpath, "model.json"), "rt") as f:
            model_config = json.load(f)

        model_type = model_config["model_type"]
        model:model_base = getattr(models, model_type)(**model_config)
        with open(os.path.join(mpath, "models", f"model_{iter}.pkl"), "rt") as f:
            state_dict = torch.load(f)
            model.load_state_dict(state_dict)

        return

    def inference(self, **kwargs):

        return

    pass


if __name__ == "__main__":
    pass

