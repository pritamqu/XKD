# -*- coding: utf-8 -*-
"""
@author: pritam sarkar
@web: www.pritamsarkar.com
"""

def get_vid_aug(name='standard', crop_size=224, num_frames=8, mode='train', aug_kwargs=None):

    from .video_augmentations import StrongTransforms2, GlobalLocalView, Basic
        
    if name == 'global_local':
        
        global_aug = aug_kwargs['global']
        local_aug = aug_kwargs['local']
        temporal_ratio = aug_kwargs['temporal_ratio']
        spatial_ratio = aug_kwargs['spatial_ratio']
        num_local_views = aug_kwargs['num_local_views']
        
        basic = Basic(crop=(crop_size, crop_size),
                      num_frames=num_frames,
                      pad_missing=True,
                      )
        
        # global; here set pad_missing to false
        global_aug = StrongTransforms2(
            crop=(crop_size,crop_size),
            num_frames=num_frames,
            mode=mode,
            **global_aug)  
        
        # local; here set pad_missing to false
        local_aug = StrongTransforms2(
            crop=(round(crop_size//spatial_ratio),round(crop_size//spatial_ratio)),
            num_frames=num_frames//temporal_ratio,
            mode=mode,
            **local_aug)
        
        augmentation = GlobalLocalView(basic, global_aug, local_aug, temporal_ratio, num_local_views)
        

    else:
        raise NotImplementedError

    return augmentation


def get_aud_aug(name='standard', audio_fps=16000, n_fft=1024, n_mels=80, duration=2, hop_length=160, mode='train', aug_kwargs=None):
    
    from .audio_augmentations import AudioPrep, GlobalLocalView, SpecAug
    
   
    if name == 'global_local':
        
        global_aug = aug_kwargs['global']
        local_aug = aug_kwargs['local']
        temporal_ratio = aug_kwargs['temporal_ratio']
        num_local_views = aug_kwargs['num_local_views']
        
        basic = AudioPrep(
            sr=audio_fps,
            trim_pad=True,
            duration=duration,
            missing_as_zero=True, 
            to_tensor=True)
        
        # global; here set trim_pad to false
        global_aug = SpecAug(
            mode=mode,
            audio_fps=audio_fps,
            n_fft=n_fft,
            n_mels=n_mels,
            hop_length=hop_length,
            # duration=duration[0],
            **global_aug) 
        
        # local; here set trim_pad to false
        local_aug = SpecAug(
            mode=mode,
            audio_fps=audio_fps,
            n_fft=n_fft,
            n_mels=n_mels,
            hop_length=hop_length,
            # duration=duration[1],
            **local_aug) 
        
        augmentation = GlobalLocalView(basic, global_aug, local_aug, temporal_ratio, num_local_views,)
        
    
    else:
        raise NotImplementedError

    return augmentation
