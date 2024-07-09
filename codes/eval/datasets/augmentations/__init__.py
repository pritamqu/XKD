# -*- coding: utf-8 -*-
"""
@author: pritam sarkar
@web: www.pritamsarkar.com
"""
from .video_augmentations import BatchMultiplier

def get_vid_aug(name='standard', crop_size=224, num_frames=8, mode='train', aug_kwargs=None, batch_multiplier=0):

    from .video_augmentations import StandardTransforms, StrongTransforms, StrongTransforms3Crop, \
        RandVisTransforms, RandVisTransforms3Crop

    if name == 'standard':
        augmentation = StandardTransforms(
            crop=(crop_size,crop_size),
            num_frames=num_frames,
            mode=mode,
            **aug_kwargs)
        
    elif name == 'strong':
        augmentation = StrongTransforms(
            crop=(crop_size,crop_size),
            num_frames=num_frames,
            mode=mode,
            **aug_kwargs) 
        
    elif name == 'strong3crop':
        augmentation = StrongTransforms3Crop(
            crop=(crop_size,crop_size),
            num_frames=num_frames,
            mode=mode,
            **aug_kwargs) 
        
    elif name == 'randaug':
        augmentation = RandVisTransforms(
            crop=(crop_size,crop_size),
            num_frames=num_frames,
            mode=mode,
            **aug_kwargs) 
        
    elif name == 'randaug3crop':
        augmentation = RandVisTransforms3Crop(
            crop=(crop_size,crop_size),
            num_frames=num_frames,
            mode=mode,
            **aug_kwargs) 
    else:
        raise NotImplementedError
        
    if batch_multiplier>0:
        return BatchMultiplier(multiplier=batch_multiplier, 
                               augmentation=augmentation)
    else:
        return augmentation


def get_aud_aug(name='standard', audio_fps=16000, n_fft=1024, n_mels=80, duration=2, hop_length=160, mode='train', aug_kwargs=None, batch_multiplier=0):
    
    from .audio_augmentations import StandardAug, StrongAug

    if name == 'standard':
        augmentation = StandardAug(
            mode=mode,
            audio_fps=audio_fps,
            n_fft=n_fft,
            n_mels=n_mels,
            hop_length=hop_length,
            duration=duration,
            **aug_kwargs)
                    
    elif name == 'strong':
        augmentation = StrongAug(
            mode=mode,
            audio_fps=audio_fps,
            n_fft=n_fft,
            n_mels=n_mels,
            hop_length=hop_length,
            duration=duration,
            **aug_kwargs) 

    else:
        raise NotImplementedError

    if batch_multiplier>0:
        return BatchMultiplier(multiplier=batch_multiplier, 
                               augmentation=augmentation)
    else:
        return augmentation
