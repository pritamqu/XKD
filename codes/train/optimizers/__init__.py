import torch
import numpy as np
from torch._six import inf
from timm.optim.optim_factory import add_weight_decay

def get_optimizer(name, model, lr=1e-3, momentum=0.9, weight_decay=0, betas=(0.9, 0.999)):

    # optimizer
    if name == 'adamw':
        # following https://github.com/facebookresearch/mae/blob/main/main_pretrain.py
        if weight_decay is None: # this is to add a different weight decay schedule
            parameters = get_params_groups(model)
        else:
            parameters = add_weight_decay(model, weight_decay)
        optimizer = torch.optim.AdamW(parameters, lr=lr, betas=betas)
    else:
        raise NotImplementedError

    return optimizer

def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [
        {'params': not_regularized, 'weight_decay': 0.}, 
        {'params': regularized}
            ]


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    # assert len(schedule) == epochs * niter_per_ep
    return schedule
