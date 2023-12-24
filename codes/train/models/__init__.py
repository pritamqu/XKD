import torch
import torch.nn as nn
from models.xkd import XKD
from models.xkd_mas import XKD_MAS
from models.xkd_mats import XKD_MATS




def get_backbone(backbone, castrate=True):
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