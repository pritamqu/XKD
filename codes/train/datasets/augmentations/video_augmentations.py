# -*- coding: utf-8 -*-
"""
@author: pritam sarkar
@web: www.pritamsarkar.com
"""

import torch
import random
from datasets.videotransforms import video_transforms, volume_transforms, tensor_transforms

class StrongTransforms2(object):
    """
    a series of strong transformation on one clip
    all transformations are temporarily consistent.
    if p_ = 0, aug not applied at all, if 1 always applied
    """

    def __init__(self,
                 crop=(224, 224),
                 color=(0.4, 0.4, 0.4, 0.2),
                 crop_scale=(0.2, 1.),
                 # cutout_size=20, # max of 10x10 sq block
                 # num_of_cutout=1,
                 p_flip=0.5,
                 p_gray=0.2,
                 p_blur=0.0,
                 # p_cutout=1.0,
                 mode='train',
                 normalize=True,
                 totensor=True,
                 num_frames=8,
                 pad_missing=False,
                 ):

        self.crop = crop
        self.mode = mode
        self.num_frames = num_frames
        self.pad_missing = pad_missing
        apply_hf = False if p_flip == 0 else True
        apply_cj = False if all(v == 0 for v in color) else True
        apply_rg = False if p_gray == 0 else True
        apply_gb = False if p_blur == 0 else True

        if normalize:
            assert totensor

        # for training
        train_transforms = [video_transforms.RandomResizedCrop(crop, scale=crop_scale)]
        if apply_hf>0:
            train_transforms.append(video_transforms.RandomHorizontalFlip(p_flip))
        if apply_cj:
            train_transforms.append(video_transforms.ColorJitter(*color))
        if apply_rg:
            train_transforms.append(video_transforms.RandomGray(p_gray))
        if apply_gb:
            train_transforms.append(video_transforms.RandomGaussianBlur(kernel_size=crop[0]//20*2+1, sigma=(0.1, 2.0), p=p_blur))

        # for validation
        val_transforms = [
            video_transforms.Resize(int(crop[0]/0.875)),
            video_transforms.CenterCrop(crop),
        ]

        def _prepare_transformations(transforms):
            if totensor:
                transforms += [volume_transforms.ClipToTensor()]
                if normalize:
                    transforms += [tensor_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
            transform = video_transforms.Compose(transforms)
            return transform

        self.train_transforms = _prepare_transformations(train_transforms)
        self.val_transforms = _prepare_transformations(val_transforms)
        ## applied only in training and it is a tensor level operation
        # self.cutout_transforms = video_transforms.Cutout(p_cutout, cutout_size, num_of_cutout, value=None) # set value to None if you want to apply mask with mean value, else set to 0

    def _if_pad_missing(self, frames):
        while True:
            n_missing = self.num_frames - frames.shape[1]
            if n_missing > 0:
                frames = torch.cat((frames, frames[:, :int(n_missing)]), 1)
            else:
                break

        return frames

    def __call__(self, frames):
        if self.mode=='train':
            frames=self.train_transforms(frames)
            # frames=self.cutout_transforms(frames)
        elif self.mode=='val':
            frames=self.val_transforms(frames)
        else:
            raise NotImplementedError(f'transformation mode: {self.mode} is not available')

        if self.pad_missing:
            frames = self._if_pad_missing(frames)

        return frames
    
    
    
class Basic(object):
    """
    basic processing before any augmentations; on PIL images;
    Simply resize and pad missing frames;
    this will be followed by creating a global local view
    """
    def __init__(self,
                 crop=(224, 224),
                 num_frames=8,
                 pad_missing=True,
                 ):

        self.num_frames = num_frames
        self.pad_missing = pad_missing
        self.resize = video_transforms.Resize(crop[0])

    def _if_pad_missing(self, frames):
        while True:
            n_missing = self.num_frames - len(frames)
            if n_missing > 0:
                frames.append(frames[:, :int(n_missing)])
            else:
                break

        return frames

    def __call__(self, frames):
        frames=self.resize(frames)
        if self.pad_missing:
            frames = self._if_pad_missing(frames)
        return frames

class GlobalLocalView(object):
    
    def __init__(self, 
                 basic_transforms,
                 global_transforms, 
                 local_transforms,
                 temporal_ratio,
                 num_local_views,
                 ):
        
        self.basic_transforms = basic_transforms
        self.global_transforms = global_transforms
        self.local_transforms = local_transforms
        self.temporal_ratio = temporal_ratio
        self.num_local_views = num_local_views

    def __call__(self, frames):
        
        holder = []
        frames = self.basic_transforms(frames)
        global_view = self.global_transforms(frames)
        
        holder.append(global_view)
        
        local_len = len(frames)//self.temporal_ratio
        for k in range(0, self.num_local_views):
            st = random.randint(0, len(frames)-local_len)
            local_view = frames[st: st+local_len]
            local_view = self.local_transforms(local_view)
            holder.append(local_view)

        return holder
    
    
    
    
    
    
    