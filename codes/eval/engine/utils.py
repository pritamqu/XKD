# -*- coding: utf-8 -*-
"""
@author: pritam sarkar
@web: www.pritamsarkar.com
"""

from tools import Logger, ProgressMeter, AverageMeter, accuracy
import torch
import torch.nn as nn
import os
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F 
import torch.distributed as dist
from collections import defaultdict
from timm.optim.optim_factory import add_weight_decay

########### common stuff

def set_grad(nets, requires_grad=False):
    for param in nets.parameters():
        param.requires_grad = requires_grad
            
########### finetune stuff

def warmup_cosine_scheduler(warmup_epochs, warmup_lr, num_epochs, base_lr, final_lr, iter_per_epoch):
    warmup_iter = iter_per_epoch * warmup_epochs
    warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
    decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
    cosine_lr_schedule = final_lr+0.5*(base_lr-final_lr)*(1+np.cos(np.pi*np.arange(decay_iter)/decay_iter))
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
    
    return lr_schedule

def warmup_multistep_scheduler(warmup_epochs, warmup_lr, num_epochs, base_lr, milestones, gamma, iter_per_epoch):
    warmup_iter = iter_per_epoch * warmup_epochs
    warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
    milestones = [iter_per_epoch * m for m in milestones]
    total_iter = iter_per_epoch * num_epochs
    multistep_lr_schedule = []
    lr = base_lr
    for i in range(warmup_iter, total_iter):
        if i in milestones:
            lr = lr*gamma
        multistep_lr_schedule.append(lr)
    multistep_lr_schedule = np.array(multistep_lr_schedule)
    lr_schedule = np.concatenate((warmup_lr_schedule, multistep_lr_schedule))

    return lr_schedule

def warmup_fixed_scheduler(warmup_epochs, warmup_lr, num_epochs, base_lr, iter_per_epoch):
    warmup_iter = iter_per_epoch * warmup_epochs
    warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
    
    fixed_iters = (num_epochs - warmup_epochs) * iter_per_epoch
    fixed_lr_schedule = np.ones(fixed_iters) * base_lr
    lr_schedule = np.concatenate((warmup_lr_schedule, fixed_lr_schedule))

    return lr_schedule

# def get_optimizer(name, model, lr=1e-3, momentum=0.9, weight_decay=0, betas=(0.9, 0.999)):

#     # optimizer
#     if name == 'adamw':
#         # following https://github.com/facebookresearch/mae/blob/main/main_pretrain.py
#         if weight_decay is None: # this is to add a different weight decay schedule
#             parameters = get_params_groups(model)
#         else:
#             parameters = add_weight_decay(model, weight_decay)
#         optimizer = torch.optim.AdamW(parameters, lr=lr, betas=betas)
#     else:
#         raise NotImplementedError

#     return optimizer

# def get_params_groups(model):
#     regularized = []
#     not_regularized = []
#     for name, param in model.named_parameters():
#         if not param.requires_grad:
#             continue
#         # we do not regularize biases nor Norm parameters
#         if name.endswith(".bias") or len(param.shape) == 1:
#             not_regularized.append(param)
#         else:
#             regularized.append(param)
#     return [
#         {'params': not_regularized, 'weight_decay': 0.}, 
#         {'params': regularized}
#             ]

def get_params_groups(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    num_layers = len(model.blocks) + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay
            
        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    # print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

    return list(param_groups.values())


def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name in ['cls_token', 'pos_embed']:
        return 0
    elif name.startswith('patch_embed'):
        return 0
    elif name.startswith('blocks'):
        return int(name.split('.')[1]) + 1
    else:
        return num_layers




# def get_optimizer(params, cfg, logger=None):
        
#     ## optimizer 
#     if cfg['name'] == 'sgd':
#         optimizer = torch.optim.SGD(
#             params=params,
#             lr=cfg['lr']['base_lr'],
#             momentum=cfg['momentum'],
#             weight_decay=cfg['weight_decay'],
#             nesterov=False,
#         )

#     elif cfg['name'] == 'adam':
#         optimizer = torch.optim.Adam(
#             params=params,
#             lr=cfg['lr']['base_lr'],
#             weight_decay=cfg['weight_decay'],
#             betas=cfg['betas'] if 'betas' in cfg else [0.9, 0.999]
#         )
        
#     elif cfg['name'] == 'adamw':
        

#     else:
#         raise ValueError('Unknown optimizer.')


#     ## lr scheduler 
#     if cfg['lr']['name']=='fixed':
#         scheduler = MultiStepLR(optimizer, milestones=[cfg['num_epochs']], gamma=1)
#     elif cfg['lr']['name']=='multistep':
#         scheduler = MultiStepLR(optimizer, milestones=cfg['lr']['milestones'], gamma=cfg['lr']['gamma'])
#     else:
#         raise NotImplementedError(f"{cfg['lr']['name']} is not yet implemented")
        
#     return optimizer, scheduler

def save_checkpoint(args, classifier, optimizer, epoch, name='classifier'):
    # Save checkpoint
    model_path = os.path.join(args.ckpt_dir, name + ".pth.tar")
    
    checkpoint = {'optimizer': optimizer.state_dict(), 
                'classifier': classifier.state_dict(), 
                'epoch': epoch + 1}

    torch.save(checkpoint, model_path)
    print(f"Classifier saved to {model_path}")

############## feature extraction

class Feature_Bank(object):
   
    def __init__(self, world_size, distributed, net, logger, print_freq=10, mode='vid', l2_norm=True):

        # mode = vid or aud
        self.mode = mode
        self.world_size = world_size
        self.distributed = distributed
        self.net = net
        self.logger = logger
        self.print_freq = print_freq
        self.l2_norm = l2_norm
        
    @torch.no_grad()
    def fill_memory_bank(self, data_loader):
            
        feature_bank = []
        feature_labels = []
        feature_indexs = []
        self.logger.add_line("Extracting features...")
        phase = 'test_dense' if data_loader.dataset.mode == 'video' else None
        
        for it, sample in enumerate(data_loader):
            if self.mode == 'vid':
                data = sample['frames'] 
            elif self.mode == 'aud':
                data = sample['audio']

            target = sample['label'].cuda(non_blocking=True)
            index = sample['index'].cuda(non_blocking=True)
            
            if phase == 'test_dense':
                batch_size, clips_per_sample = data.shape[0], data.shape[1]
                data = data.flatten(0, 1).contiguous()
                
            feature = self.net(data.cuda(non_blocking=True)).detach()
            if self.l2_norm:
                feature = F.normalize(feature, dim=1) # l2 normalize
            if feature.shape[0]==1:
                pass
            else:
                feature = torch.squeeze(feature)
            
            if phase == 'test_dense':
                feature = feature.view(batch_size, clips_per_sample, -1).contiguous()
                
            if self.distributed:
                # create blank tensor
                sub_feature_bank    = [torch.ones_like(feature) for _ in range(self.world_size)]
                sub_labels_bank     = [torch.ones_like(target) for _ in range(self.world_size)]
                sub_index_bank      = [torch.ones_like(index) for _ in range(self.world_size)]
                # gather from all processes
                dist.all_gather(sub_feature_bank, feature)
                dist.all_gather(sub_labels_bank, target)
                dist.all_gather(sub_index_bank, index)
                # concat them 
                sub_feature_bank = torch.cat(sub_feature_bank)
                sub_labels_bank = torch.cat(sub_labels_bank)
                sub_index_bank = torch.cat(sub_index_bank)
                # append to one bank in all processes
                feature_bank.append(sub_feature_bank.contiguous().cpu())
                feature_labels.append(sub_labels_bank.cpu())
                feature_indexs.append(sub_index_bank.cpu())
                
            else:
                
                feature_bank.append(feature.contiguous().cpu())
                feature_labels.append(target.cpu())
                feature_indexs.append(index.cpu())
            
            # print(feature.shape)
            if it%100==0:
                self.logger.add_line(f'{it} / {len(data_loader)}')
                    
        feature_bank    = torch.cat(feature_bank, dim=0)
        feature_labels  = torch.cat(feature_labels)
        feature_indexs  = torch.cat(feature_indexs)
        
        return feature_bank, feature_labels, feature_indexs
    
class Feature_Bank_AV(object):
   
    def __init__(self, world_size, distributed, vnet, anet, logger, print_freq=10, mode='both', 
                 apply_l2=True, apply_bn=False, feat_dim=768):

        # mode = vid or aud
        self.mode = mode
        self.world_size = world_size
        self.distributed = distributed
        self.vnet = vnet
        self.anet = anet
        self.logger = logger
        self.print_freq = print_freq
        self.apply_l2 = apply_l2
        self.apply_bn = apply_bn
        if self.apply_bn:
            self.vid_bn = nn.BatchNorm1d(feat_dim)
            self.aud_bn = nn.BatchNorm1d(feat_dim)
        
    @torch.no_grad()
    def fill_memory_bank(self, data_loader):
            
        vfeature_bank = []
        afeature_bank = []
        feature_labels = []
        feature_indexs = []
        self.logger.add_line("Extracting features...")
        phase = 'test_dense' if data_loader.dataset.mode == 'video' else None
        
        for it, sample in enumerate(data_loader):
            
            vdata = sample['frames'] 
            adata = sample['audio']
            
            target = sample['label'].cuda(non_blocking=True)
            index = sample['index'].cuda(non_blocking=True)
            
            if phase == 'test_dense':
                print('vdata.shape', vdata.shape)
                three_crop = True if len(vdata.shape) == 6 else False
                batch_size, clips_per_sample = vdata.shape[0], vdata.shape[1]
                _, clips_per_sample_aud = adata.shape[0], adata.shape[1]
                vdata = vdata.flatten(0, 1).contiguous()
                adata = adata.flatten(0, 1).contiguous()
                
            vfeature = self.vnet(vdata.cuda(non_blocking=True)).detach()
            afeature = self.anet(adata.cuda(non_blocking=True)).detach()
            if self.apply_l2:
                vfeature = F.normalize(vfeature, p=2, dim=1) # l2 normalize
                afeature = F.normalize(afeature, p=2, dim=1) # l2 normalize
            if self.apply_bn:
                vfeature = self.vid_bn(vfeature)
                afeature = self.aud_bn(afeature)
            vfeature = torch.squeeze(vfeature)
            afeature = torch.squeeze(afeature)
            # print(vfeature.shape, afeature.shape)
            
            if phase == 'test_dense':
                if three_crop:
                    # clips_per_sample_aud --> true num of clips per video
                    vfeature = vfeature.view(batch_size, clips_per_sample_aud, clips_per_sample//clips_per_sample_aud, -1).contiguous()
                    vfeature = vfeature.mean(dim=2)
                else:
                    vfeature = vfeature.view(batch_size, clips_per_sample, -1).contiguous()
                    
                afeature = afeature.view(batch_size, clips_per_sample_aud, -1).contiguous()
                
            if self.distributed:
                # create blank tensor
                vsub_feature_bank    = [torch.ones_like(vfeature) for _ in range(self.world_size)]
                asub_feature_bank    = [torch.ones_like(afeature) for _ in range(self.world_size)]
                sub_labels_bank     = [torch.ones_like(target) for _ in range(self.world_size)]
                sub_index_bank      = [torch.ones_like(index) for _ in range(self.world_size)]
                # gather from all processes
                dist.all_gather(vsub_feature_bank, vfeature)
                dist.all_gather(asub_feature_bank, afeature)
                dist.all_gather(sub_labels_bank, target)
                dist.all_gather(sub_index_bank, index)
                # concat them 
                vsub_feature_bank = torch.cat(vsub_feature_bank)
                asub_feature_bank = torch.cat(asub_feature_bank)
                sub_labels_bank = torch.cat(sub_labels_bank)
                sub_index_bank = torch.cat(sub_index_bank)
                # append to one bank in all processes
                vfeature_bank.append(vsub_feature_bank.contiguous().cpu())
                afeature_bank.append(asub_feature_bank.contiguous().cpu())
                feature_labels.append(sub_labels_bank.cpu())
                feature_indexs.append(sub_index_bank.cpu())
                
            else:
                
                vfeature_bank.append(vfeature.contiguous().cpu())
                afeature_bank.append(afeature.contiguous().cpu())
                feature_labels.append(target.cpu())
                feature_indexs.append(index.cpu())
            
            if it%100==0:
                self.logger.add_line(f'{it} / {len(data_loader)}')
                    
        vfeature_bank    = torch.cat(vfeature_bank, dim=0)
        afeature_bank    = torch.cat(afeature_bank, dim=0)
        feature_labels  = torch.cat(feature_labels)
        feature_indexs  = torch.cat(feature_indexs)
        
        return vfeature_bank, afeature_bank, feature_labels, feature_indexs
        

def average_features(
    features, 
    labels, 
    indices, 
    logger=None,
    norm_feats=True,
    ):
    
    # src: https://github.com/facebookresearch/selavi/
    
    feat_dict = defaultdict(list)
    label_dict = defaultdict(list)
    for i in range(len(features)):
        if norm_feats:
            v = features[i]
            feat = v / np.sqrt(np.sum(v**2))
        else:
            feat = features[i]
        label = labels[i]
        idx = indices[i]
        feat_dict[idx].append(feat)
        label_dict[idx].append(label)
        print(f'{i} / {len(features)}', end='\r')

    avg_features, avg_indices, avg_labels = [], [], []
    num_features = 0
    for idx in feat_dict:
        stcked_feats = np.stack(feat_dict[idx]).squeeze(axis=0)
        feat = np.mean(stcked_feats, axis=0)
        vid_ix_feat_len = stcked_feats.shape[0]
        num_features += vid_ix_feat_len
        label = label_dict[idx][0]
        avg_features.append(feat)
        avg_indices.append(idx)
        avg_labels.append(label)
    avg_features = np.stack(avg_features, axis=0)
    avg_indices = np.stack(avg_indices, axis=0)
    avg_labels = np.stack(avg_labels, axis=0)

    return avg_features, avg_labels, avg_indices


# class Classifier(nn.Module):
#     " for fixed feature representation"
#     def __init__(self, n_classes, feat_dim, l2_norm=False, use_bn=False, use_dropout=False, dropout=0.5):
#         super(Classifier, self).__init__()
#         self.use_bn = use_bn
#         self.l2_norm = l2_norm
#         self.use_dropout = use_dropout
#         if use_bn:
#             self.bn = nn.BatchNorm1d(feat_dim)
#         if use_dropout:
#             self.dropout = nn.Dropout(dropout)
#         self.classifier = nn.Linear(feat_dim, n_classes)

#     def forward(self, x):
#         with torch.no_grad():
#             if self.use_dropout:
#                 x = self.dropout(x)
#             if self.l2_norm:
#                 x = nn.functional.normalize(x, p=2, dim=-1)
#             x = x.view(x.shape[0], -1).contiguous().detach()
#         if self.use_bn:
#             x = self.bn(x)
#         return self.classifier(x)
    
    
class Classifier(nn.Module):
    "classifier head"
    def __init__(self, num_classes, feat_dim, l2_norm=False, use_bn=False, use_dropout=False, dropout=0.5):
        super(Classifier, self).__init__()
        self.use_bn = use_bn
        self.l2_norm = l2_norm
        self.use_dropout = use_dropout
        if use_bn:
            self.bn = nn.BatchNorm1d(feat_dim)
            self.bn.weight.data.fill_(1)
            self.bn.bias.data.zero_()
        if use_dropout:
            self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feat_dim, num_classes)
        self._initialize_weights(self.classifier)

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)
                
    def forward(self, x):
        x = x.squeeze()
        # x = x.view(x.shape[0], -1)
        if self.l2_norm:
            x = nn.functional.normalize(x, p=2, dim=-1)        
        if self.use_bn:
            x = self.bn(x)
        if self.use_dropout:
            x = self.dropout(x)
        return self.classifier(x)