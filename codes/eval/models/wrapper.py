# -*- coding: utf-8 -*-
"""
@author: pritam sarkar
@web: www.pritamsarkar.com
"""

import torch.nn as nn
import torch

class LinearClassifier(nn.Module):
    def __init__(self, dim, use_bn=False, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()
        self.bn = nn.BatchNorm1d(dim, affine=False, eps=1e-6)
        self.use_bn = use_bn

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        if self.use_bn:
            x = self.bn(x)
        return self.linear(x)

class FullFtWrapper(nn.Module):
    def __init__(self, backbone, classifier, feat_op='pool', use_amp=True):
        super(FullFtWrapper, self).__init__()
        self.feat_op = feat_op
        self.backbone = backbone
        self.classifier = classifier
        self.use_amp = use_amp
        
    def fwd(self, x):
        # pass the backbone
        x = self.backbone(x)
        if self.feat_op =='pool':
            if self.backbone.apply_cls_token:
                x = x[:, 1:, :].mean(dim=1)
            else:
                x = x.mean(dim=1)
        elif self.feat_op =='cls':
            assert self.backbone.apply_cls_token
            x = x[:, 0]
        else:
            raise ValueError(f'feat_op should be either pool or cls; given {self.feat_op}')
            
        # pass head
        x = self.classifier(x)
        return x

    def forward(self, x):
        if self.use_amp:
            with torch.cuda.amp.autocast():
                return self.fwd(x)
        else:
            return self.fwd(x)
            

class LinearFtWrapper(nn.Module):
    def __init__(self, backbone, classifier, feat_op='pool', use_amp=False):
        super(LinearFtWrapper, self).__init__()
        self.feat_op = feat_op
        self.backbone = backbone
        self.classifier = classifier
        self.use_amp = use_amp
        
    @torch.no_grad()
    def fwd(self, x):
        # pass the backbone
        x = self.backbone(x)
        if self.feat_op =='pool':
            if self.backbone.apply_cls_token:
                x = x[:, 1:, :].mean(dim=1)
            else:
                x = x.mean(dim=1)
        elif self.feat_op =='cls':
            assert self.backbone.apply_cls_token
            x = x[:, 0]
        else:
            raise ValueError(f'feat_op should be either pool or cls; given {self.feat_op}')
            
        return x

    def forward(self, x):
        if self.use_amp:
            with torch.cuda.amp.autocast():
                x = self.fwd(x)
                x = self.classifier(x)
                return x
        else:
            x = self.fwd(x)
            x = self.classifier(x)
            return x
        
class SVMWrapper(nn.Module):
    def __init__(self, backbone, feat_op='pool', use_amp=False):
        super(SVMWrapper, self).__init__()
        self.feat_op = feat_op
        self.backbone = backbone
        self.use_amp = use_amp
        
    def forward(self, x):
        with torch.no_grad():
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    x = self.backbone(x, self.feat_op)
                return x
            else:
                x = self.backbone(x, self.feat_op)
                return x