# -*- coding: utf-8 -*-
"""
@author: pritam sarkar
@web: www.pritamsarkar.com
"""

import torch
import torch.nn as nn
from .modules import *
from .wrapper import FullFtWrapper, LinearFtWrapper, LinearClassifier, SVMWrapper
import torch.nn as nn

def get_backbone(backbone):
    backbone = eval(f"{backbone}")
    return backbone

def get_model(model_cfg):
    return eval(model_cfg['name'])(**model_cfg['kwargs'])


def weight_reset(m):
    import torch.nn as nn
    if (
        isinstance(m, nn.Conv1d)
        or isinstance(m, nn.Conv2d)
        or isinstance(m, nn.Linear)
        or isinstance(m, nn.Conv3d)
        or isinstance(m, nn.ConvTranspose1d)
        or isinstance(m, nn.ConvTranspose2d)
        or isinstance(m, nn.ConvTranspose3d)
        or isinstance(m, nn.BatchNorm1d)
        or isinstance(m, nn.BatchNorm2d)
        or isinstance(m, nn.BatchNorm3d)
        or isinstance(m, nn.GroupNorm)
    ):
        m.reset_parameters()

def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False

