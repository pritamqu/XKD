# -*- coding: utf-8 -*-
"""
@author: pritam sarkar
@web: www.pritamsarkar.com
"""

import torch
import sklearn
from datasets.audiotransforms.wav_aug import AudioPrep, VolJitter 
from datasets.audiotransforms.to_spec import MelSpectrogramLibrosa
from datasets.audiotransforms.spec_aug import TimeWarp, MaskAlongAxis, RandomResizeCrop
import random


class SpecAug(object):
    
    def __init__(self, 
                     mode='train',
                     audio_fps=16000, n_fft=1024, n_mels=80, duration=2, hop_length=160, f_min=0, f_max=None,
                     vol=0.2, wrap_window=50,
                     virtual_crop_scale=(1.0, 1.5), freq_scale=(0.6, 1.5), time_scale=(0.6, 1.5),
                     # fmask_len=(0, 30), tmask_len=(0, 30), num_fmask=2, num_tmask=2, 
                     voljitter=False, timewarp=False, randcrop=False,
                     normalize=True,
                     trim_pad=True,
                     ):

        self.mode = mode
        self.mean, self.std = -5.3688107, 4.410007
        # self.preprocessing = AudioPrep(
        #     sr=audio_fps,
        #     trim_pad=trim_pad,
        #     duration=duration,
        #     missing_as_zero=True, 
        #     to_tensor=True)
        self.to_melspecgram = MelSpectrogramLibrosa(
            fs=audio_fps,
            n_fft=n_fft,
            shift=hop_length,
            n_mels=n_mels,
            fmin=f_min,
            fmax=f_max,
        ) 
        
        self.normalize = normalize
        self.voljitter = VolJitter(vol) if voljitter else None
        # self.fmask = MaskAlongAxis(mask_width_range=fmask_len, num_mask=num_fmask, dim='freq', replace_with_zero=False) if fmask else None
        # self.tmask = MaskAlongAxis(mask_width_range=tmask_len, num_mask=num_tmask, dim='time', replace_with_zero=False) if tmask else None
        self.timewarp = TimeWarp(window=wrap_window) if timewarp else None
        self.randcrop = RandomResizeCrop(virtual_crop_scale=virtual_crop_scale, freq_scale=freq_scale, time_scale=time_scale) if randcrop else None

    def __call__(self, out):
        if self.mode=='val' or self.mode=='test':
            # out = wav[0]
            out = (self.to_melspecgram(out) + torch.finfo().eps).log().unsqueeze(0) # (1 x mel x timesteps) = (1 x 80 x 201)
            # normalize with pre-computed mean and std
            if self.normalize:
                out = (out - self.mean)/self.std
            return out
                
        elif self.mode=='train':
            # out = wav[0]
            if self.voljitter is not None:
                out = torch.from_numpy(self.voljitter(out.numpy()))
            # out = out[0] # (N,)
            out = (self.to_melspecgram(out) + torch.finfo().eps).log().unsqueeze(0) # (1 x mel x timesteps) = (1 x 80 x 201)
            # normalize with pre-computed mean and std
            if self.normalize:
                out = (out - self.mean)/self.std            
            # spec level augs
            if self.timewarp is not None:
                out = self.timewarp.time_warp(out.transpose(2, 1)).transpose(2, 1) # takes input as spec: (Batch, Length, Freq)
            if self.randcrop is not None:
                out = self.randcrop(out)
                
        return out

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

    def __call__(self, wav):
        
        holder = []
        wav = self.basic_transforms(wav)
        wav = wav[0] # wav
        
        # global
        global_view = self.global_transforms(wav)
        holder.append(global_view)
        
        local_len = len(wav)//self.temporal_ratio
        for k in range(0, self.num_local_views):
            st = random.randint(0, (len(wav)-local_len)//100)*100 # with a step of 100
            local_view = wav[st: st+local_len]
            local_view = self.local_transforms(local_view)
            holder.append(local_view)
        
        return holder
    
    