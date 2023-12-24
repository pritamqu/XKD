# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 17:09:40 2022

@author: pritam
"""

# Most of the codes are borrowed from DINO and iBOT
# https://github.com/facebookresearch/dino
# https://github.com/bytedance/ibot/

import torch
import torch.nn as nn
import torch.nn.functional as F 
from timm.models.layers import trunc_normal_
import torch.distributed as dist

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

class CSyncBatchNorm(nn.SyncBatchNorm):
    def __init__(self,
                 *args,
                 with_var=False,
                 **kwargs):
        super(CSyncBatchNorm, self).__init__(*args, **kwargs)
        self.with_var = with_var

    def forward(self, x):
        # center norm
        self.training = False
        if not self.with_var:
            self.running_var = torch.ones_like(self.running_var)
        normed_x = super(CSyncBatchNorm, self).forward(x)
        # udpate center
        self.training = True
        _ = super(CSyncBatchNorm, self).forward(x)
        return normed_x

class PSyncBatchNorm(nn.SyncBatchNorm):
    def __init__(self,
                 *args,
                 bunch_size,
                 **kwargs):
        procs_per_bunch = min(bunch_size, get_world_size())
        assert get_world_size() % procs_per_bunch == 0
        n_bunch = get_world_size() // procs_per_bunch
        #
        ranks = list(range(get_world_size()))
        print('---ALL RANKS----\n{}'.format(ranks))
        rank_groups = [ranks[i*procs_per_bunch: (i+1)*procs_per_bunch] for i in range(n_bunch)]
        print('---RANK GROUPS----\n{}'.format(rank_groups))
        process_groups = [torch.distributed.new_group(pids) for pids in rank_groups]
        bunch_id = get_rank() // procs_per_bunch
        process_group = process_groups[bunch_id]
        print('---CURRENT GROUP----\n{}'.format(process_group))
        super(PSyncBatchNorm, self).__init__(*args, process_group=process_group, **kwargs)

class CustomSequential(nn.Sequential):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)

    def forward(self, input):
        for module in self:
            dim = len(input.shape)
            if isinstance(module, self.bn_types) and dim > 2:
                perm = list(range(dim - 1)); perm.insert(1, dim - 1)
                inv_perm = list(range(dim)) + [1]; inv_perm.pop(1)
                input = module(input.permute(*perm)).permute(*inv_perm)
            else:
                input = module(input)
        return input
    
class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, norm=None, act='gelu', last_norm=None, 
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256, norm_last_layer=True, **kwargs):
        """
        copied from https://github.com/facebookresearch/dino/blob/main/vision_transformer.py#L257
        """
        
        super().__init__()
        self.out_dim = out_dim
        norm = self._build_norm(norm, hidden_dim)
        last_norm = self._build_norm(last_norm, out_dim, affine=False, **kwargs)
        act = self._build_act(act)
        
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            if bottleneck_dim > 0:
                self.mlp = nn.Linear(in_dim, bottleneck_dim)
            else:
                self.mlp = nn.Linear(in_dim, out_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if norm is not None:
                layers.append(norm)
            layers.append(act)
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if norm is not None:
                    layers.append(norm)
                layers.append(act)
            if bottleneck_dim > 0:
                layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            else:
                layers.append(nn.Linear(hidden_dim, out_dim))
            self.mlp = CustomSequential(*layers)
        self.apply(self._init_weights)
        
        if bottleneck_dim > 0:
            self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
            self.last_layer.weight_g.data.fill_(1)
            if norm_last_layer:
                self.last_layer.weight_g.requires_grad = False
        else:
            self.last_layer = None

        self.last_norm = last_norm

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if len(x.shape) == 3: # just take the cls
            x = x [:, 0]
            
        x = self.mlp(x)
        if self.last_layer is not None:
            x = nn.functional.normalize(x, dim=-1, p=2)
            x = self.last_layer(x)
        if self.last_norm is not None:
            x = self.last_norm(x)
        return x

    def _build_norm(self, norm, hidden_dim, **kwargs):
        if norm == 'bn':
            norm = nn.BatchNorm1d(hidden_dim, **kwargs)
        elif norm == 'syncbn':
            norm = nn.SyncBatchNorm(hidden_dim, **kwargs)
        elif norm == 'csyncbn':
            norm = CSyncBatchNorm(hidden_dim, **kwargs)
        elif norm == 'psyncbn':
            norm =  PSyncBatchNorm(hidden_dim, **kwargs)
        elif norm == 'ln':
            if 'affine' in kwargs:
                elementwise_affine = kwargs.pop('affine')
                norm = nn.LayerNorm(hidden_dim, elementwise_affine=elementwise_affine)
            else:
                norm = nn.LayerNorm(hidden_dim)
        else:
            assert norm is None, "unknown norm type {}".format(norm)
        return norm

    def _build_act(self, act):
        if act == 'relu':
            act = nn.ReLU()
        elif act == 'gelu':
            act = nn.GELU()
        else:
            assert False, "unknown act type {}".format(act)
        return act

        
class MultiViewWrapper(nn.Module):
    """
    copied from iBOT: https://github.com/bytedance/ibot/blob/da316d82636a7a7356835ef224b13d5f3ace0489/utils.py#L648
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, backbone, head=None):
        super(MultiViewWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        if head is None:
            self.head = nn.Identity()
        else:
            self.head = head

    def forward(self, x, mask=None, return_backbone_feat=False, 
                **kwargs):
        # convert to list
        if not isinstance(x, list):
            x = [x]
            mask = [mask] if mask is not None else None
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx = 0
        for end_idx in idx_crops:
            inp_x = torch.cat(x[start_idx: end_idx])

            if mask is not None:
                inp_m = torch.cat(mask[start_idx: end_idx])
                kwargs.update(dict(mask=inp_m))

            _out = self.backbone(inp_x, **kwargs)
            if start_idx == 0:
                output = _out
            else:
                output = torch.cat((output, _out))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        output_ = self.head(output)
        if return_backbone_feat:
            return output, output_
        return output_
    
    def forward_last_layer_attn(self, x, mask=None, 
                                pass_patch_mean=False, 
                                cls_token_present=False,
                                return_last_layer_attn=False, 
                                **kwargs):
        # convert to list
        if not isinstance(x, list):
            x = [x]
            mask = [mask] if mask is not None else None
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx = 0
        for end_idx in idx_crops:
            inp_x = torch.cat(x[start_idx: end_idx])

            if mask is not None:
                inp_m = torch.cat(mask[start_idx: end_idx])
                kwargs.update(dict(mask=inp_m))
            
            if return_last_layer_attn:
                attn, _out = self.backbone.forward_last_layer_attn(inp_x, **kwargs)
            else:
                _out = self.backbone(inp_x, **kwargs)
            if start_idx == 0:
                if return_last_layer_attn:
                    attentions = attn
                output = _out
            else:
                if return_last_layer_attn:
                    attentions = torch.cat((attentions, attn))
                output = torch.cat((output, _out))
            start_idx = end_idx
        
        # Run the head forward on the concatenated features.
        if pass_patch_mean:
            if cls_token_present:
                output_ = self.head(output[:, 1:, ::].mean(dim=1))
            else:
                output_ = self.head(output.mean(dim=1))
        else:
            output_ = self.head(output) # take cls in next step

        if return_last_layer_attn:
            return attentions, output_
        # if return_backbone_feat:
        #     return output, output_
        return output_