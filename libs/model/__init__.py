from .model import Model
import torch
from torch import nn
from libs.utils.comm import get_world_size


def build_model(cfg):
    if get_world_size() == 1:
        norm_layer = nn.BatchNorm2d
    else:
        norm_layer = nn.BatchNorm2d
    model = Model(
        cfg,
        norm_layer=norm_layer
    )
    return model
