# -*- coding: utf-8 -*-
"""
@author: pritam sarkar
@web: www.pritamsarkar.com
"""

import torch
from datasets.videotransforms import video_transforms, volume_transforms, tensor_transforms
from datasets.videotransforms import video_transforms2



class RandVisTransforms(object):
    """ 
    a series of visual transformation
    """

    def __init__(self,
                  auto_augment=None, 
                  crop=(224, 224),
                  hflip=True,
                  min_area = [0.08, 1.0],
                  aspect_ratio = [0.75, 1.3333],
                  train_interpolation='bilinear', 
                  rand_erase = None, # pass None or dict of arguments
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
        self.hflip = hflip
        self.min_area = min_area
        self.asp = aspect_ratio
        
        if normalize:
            assert totensor
            
        def _prepare_transformations(transforms):
            if totensor:
                transforms += [volume_transforms.ClipToTensor()]
                if normalize:
                    transforms += [tensor_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
            transform = video_transforms.Compose(transforms) # C, T, H, W
            return transform

        if mode=='train':
            self.erase_transform = None
            # for training 
            transforms = video_transforms2.create_random_augment(
                input_size=crop,
                auto_augment=auto_augment,
                interpolation=train_interpolation,
            )
            self.transforms = _prepare_transformations(transforms)
            if rand_erase is not None:
                self.erase_transform = video_transforms2.RandomErasing(
                    **rand_erase,
                    device="cpu",
                )
            
            
        elif mode in ['val', 'test']:
            # for validation 
            transforms = [
                video_transforms.Resize(int(crop[0]/0.875)),
                video_transforms.CenterCrop(crop),
            ]
            self.transforms = _prepare_transformations(transforms)
        else:
            raise NotImplementedError(f'transformation mode: {mode} is not available')
        
        

    def _if_pad_missing(self, frames):
        while True:
            n_missing = self.num_frames - frames.shape[1]
            if n_missing > 0:
                frames = torch.cat((frames, frames[:, :int(n_missing)]), 1)
            else:
                break

        return frames
        
    def __call__(self, frames):
        
        frames=self.transforms(frames) # C T H W
        
        if self.mode=='train':
            frames = video_transforms2.spatial_sampling(
                frames,
                spatial_idx=-1,
                min_scale=256,
                max_scale=320,
                crop_size=self.crop[0],
                random_horizontal_flip=self.hflip,
                inverse_uniform_sampling=False,
                aspect_ratio=self.asp,
                scale=self.min_area,
                motion_shift=False
                ) # C T H W
            
            if self.erase_transform is not None:
                frames = frames.permute(1, 0, 2, 3)
                frames = self.erase_transform(frames)
                frames = frames.permute(1, 0, 2, 3)
        
        if self.pad_missing:
            frames = self._if_pad_missing(frames)

        return frames

class RandVisTransforms3Crop(object):
    """ 
    a series of visual transformation
    adding 3 crop for val or test
    """

    def __init__(self,
                 auto_augment=None, 
                 crop=(224, 224),
                 hflip=True,
                 min_area = [0.08, 1.0],
                 aspect_ratio = [0.75, 1.3333],
                 train_interpolation='bilinear', 
                 rand_erase = None, # pass None or dict of arguments
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
        self.hflip = hflip
        self.min_area = min_area
        self.asp = aspect_ratio
        
        if normalize:
            assert totensor
            
        def _prepare_transformations(transforms):
            if totensor:
                transforms += [volume_transforms.ClipToTensor()]
                if normalize:
                    transforms += [tensor_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
            transform = video_transforms.Compose(transforms) # C, T, H, W
            return transform

        if mode=='train':
            self.erase_transform = None
            # for training 
            transforms = video_transforms2.create_random_augment(
                input_size=crop,
                auto_augment=auto_augment,
                interpolation=train_interpolation,
            )
            self.transforms = _prepare_transformations(transforms)
            if rand_erase is not None:
                self.erase_transform = video_transforms2.RandomErasing(
                    **rand_erase,
                    device="cpu",
                )
            
            
        elif mode in ['val', 'test']:
            # for validation 
            transforms = [
                # video_transforms.Resize(int(crop[0]/0.875)),
                # video_transforms.CenterCrop(crop), # will do 3 crop later
            ]
            self.transforms = _prepare_transformations(transforms)
        else:
            raise NotImplementedError(f'transformation mode: {mode} is not available')
        
        

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
            frames=self.transforms(frames) # C T H W
            frames = video_transforms2.spatial_sampling(
                frames,
                spatial_idx=-1,
                min_scale=256,
                max_scale=320,
                crop_size=self.crop[0],
                random_horizontal_flip=self.hflip,
                inverse_uniform_sampling=False,
                aspect_ratio=self.asp,
                scale=self.min_area,
                motion_shift=False
                ) # C T H W
            
            if self.erase_transform is not None:
                frames = frames.permute(1, 0, 2, 3)
                frames = self.erase_transform(frames)
                frames = frames.permute(1, 0, 2, 3)
                
            if self.pad_missing:
                frames = self._if_pad_missing(frames)
                
        else: # for test and val
            frames=self.transforms(frames) # resized and converted to tensor
            frames, _ = video_transforms2.random_short_side_scale_jitter(
                frames, self.crop[0], self.crop[0]
            )
            
            if self.pad_missing:
                frames = self._if_pad_missing(frames)
                
            frames = torch.stack([video_transforms2.uniform_crop(frames, self.crop[0], 0)[0], 
                        video_transforms2.uniform_crop(frames, self.crop[0], 1)[0],
                        video_transforms2.uniform_crop(frames, self.crop[0], 2)[0],
                        ])
            

        return frames


class BatchMultiplier(object):
    """ perform multiplier times augmentation on the loaded data
    doing this to reduce data loading time.
    """
    
    def __init__(self, 
                 multiplier=2, 
                 augmentation=None):
        
        self.multiplier = multiplier
        self.augmentation = augmentation
        
    def __call__(self, x):
        stack = []
        for k in range(self.multiplier):
            stack.append(self.augmentation(x))
            
        return torch.stack(stack)
        

class StrongTransforms3Crop(object):
    """ 
    a series of strong transformation on one clip
    all transformations are temporarily consistent.
    if p_ = 0, aug not applied at all, if 1 always applied
    """

    def __init__(self,
                 crop=(224, 224),
                 color=(0.4, 0.4, 0.4, 0.2),
                 min_area=[0.08, 1],
                 cutout_size=20, # max of 10x10 sq block
                 num_of_cutout=1,
                 p_flip=0.5,
                 p_gray=0.2,
                 p_blur=0.0,
                 p_cutout=1.0, 
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
        if normalize:
            assert totensor
            
        # for training 
        train_transforms = [
            video_transforms.RandomResizedCrop(crop, scale=min_area),
            video_transforms.RandomHorizontalFlip(p_flip),
            video_transforms.ColorJitter(*color),
            video_transforms.RandomGray(p_gray),
            video_transforms.RandomGaussianBlur(kernel_size=crop[0]//20*2+1, sigma=(0.1, 2.0), p=p_blur),
        ]

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
        self.cutout_transforms = video_transforms.Cutout(p_cutout, cutout_size, num_of_cutout, value=None) # set value to None if you want to apply mask with mean value, else set to 0

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
            frames=self.cutout_transforms(frames)
            if self.pad_missing:
                frames = self._if_pad_missing(frames)
                
        elif self.mode=='val':
            frames=self.val_transforms(frames) # resized and converted to tensor
            frames, _ = video_transforms2.random_short_side_scale_jitter(
                frames, self.crop[0], self.crop[0]
            )
            
            if self.pad_missing:
                frames = self._if_pad_missing(frames)
                
            frames = torch.stack([video_transforms2.uniform_crop(frames, self.crop[0], 0)[0], 
                        video_transforms2.uniform_crop(frames, self.crop[0], 1)[0],
                        video_transforms2.uniform_crop(frames, self.crop[0], 2)[0],
                        ])            
        else:
            raise NotImplementedError(f'transformation mode: {self.mode} is not available')
            
        return frames

class StrongTransforms(object):
    """ 
    a series of strong transformation on one clip
    all transformations are temporarily consistent.
    if p_ = 0, aug not applied at all, if 1 always applied
    """

    def __init__(self,
                 crop=(224, 224),
                 color=(0.4, 0.4, 0.4, 0.2),
                 min_area=[0.08, 1],
                 cutout_size=20, # max of 10x10 sq block
                 num_of_cutout=1,
                 p_flip=0.5,
                 p_gray=0.2,
                 p_blur=0.0,
                 p_cutout=1.0, 
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
        if normalize:
            assert totensor
            
        # for training 
        train_transforms = [
            video_transforms.RandomResizedCrop(crop, scale=min_area),
            video_transforms.RandomHorizontalFlip(p_flip),
            video_transforms.ColorJitter(*color),
            video_transforms.RandomGray(p_gray),
            video_transforms.RandomGaussianBlur(kernel_size=crop[0]//20*2+1, sigma=(0.1, 2.0), p=p_blur),
        ]

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
        self.cutout_transforms = video_transforms.Cutout(p_cutout, cutout_size, num_of_cutout, value=None) # set value to None if you want to apply mask with mean value, else set to 0

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
            frames=self.cutout_transforms(frames)
        elif self.mode=='val':
            frames=self.val_transforms(frames)
        else:
            raise NotImplementedError(f'transformation mode: {self.mode} is not available')
            
        if self.pad_missing:
            frames = self._if_pad_missing(frames)

        return frames
    
    
    
class StandardTransforms(object):
    """ 
    a series of standard transformation on one clip
    all temporarily consistent transformations
    """
    def __init__(self,
                 crop=(224, 224),
                 color=(0.4, 0.4, 0.4, 0.2),
                 min_area=0.7,
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
        if normalize:
            assert totensor

        val_transforms = [
            video_transforms.Resize(int(crop[0]/0.875)),
            video_transforms.CenterCrop(crop),
        ]
        train_transforms = [
            video_transforms.RandomResizedCrop(crop, scale=(min_area, 1.)),
            video_transforms.RandomHorizontalFlip(), 
            video_transforms.ColorJitter(*color),
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
        elif self.mode=='val':
            frames=self.val_transforms(frames)
        else:
            raise NotImplementedError(f'transformation mode: {self.mode} is not available')
            
        if self.pad_missing:
            frames = self._if_pad_missing(frames)

        return frames    
    
class StandardTransforms3Crop(object):
    """ 
    a series of visual transformation
    adding 3 crop for val or test
    same style w/o random aug and erasing
    """

    def __init__(self,
                 crop=(224, 224),
                 color=(0.4, 0.4, 0.4, 0.2),
                 hflip=True,
                 min_area = [0.2, 0.766],
                 aspect_ratio = [0.75, 1.3333],
                 mode='train', 
                 normalize=True,
                 totensor=True,
                 inverse_uniform_sampling=False,
                 num_frames=8,
                 pad_missing=False,
                 ):
       
        self.crop = crop
        self.mode = mode
        self.num_frames = num_frames
        self.pad_missing = pad_missing
        self.hflip = hflip
        self.min_area = min_area
        self.asp = aspect_ratio
        self.inverse_uniform_sampling=inverse_uniform_sampling
        
        if normalize:
            assert totensor
            
        def _prepare_transformations(transforms):
            if totensor:
                transforms += [volume_transforms.ClipToTensor()]
                if normalize:
                    transforms += [tensor_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
            transform = video_transforms.Compose(transforms) # C, T, H, W
            return transform

        if mode=='train':
            transforms =[video_transforms.ColorJitter(*color)
                         ]
            self.transforms = _prepare_transformations(transforms)           
            
        elif mode in ['val', 'test']:
            transforms = [
                            ]
            self.transforms = _prepare_transformations(transforms)
        else:
            raise NotImplementedError(f'transformation mode: {mode} is not available')
        
        

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
            frames=self.transforms(frames) # C T H W
            frames = video_transforms2.spatial_sampling(
                frames,
                spatial_idx=-1,
                min_scale=256, # this is ignored when scale and asp present
                max_scale=320, # this is ignored when scale and asp present
                crop_size=self.crop[0],
                random_horizontal_flip=self.hflip,
                inverse_uniform_sampling=self.inverse_uniform_sampling,
                aspect_ratio=self.asp,
                scale=self.min_area,
                motion_shift=False
                ) # C T H W
                
            if self.pad_missing:
                frames = self._if_pad_missing(frames)
                
        else: # for test and val
            frames=self.transforms(frames) # resized and converted to tensor
            frames, _ = video_transforms2.random_short_side_scale_jitter(
                frames, self.crop[0], self.crop[0]
            )
            
            if self.pad_missing:
                frames = self._if_pad_missing(frames)
                
            frames = torch.stack([video_transforms2.uniform_crop(frames, self.crop[0], 0)[0], 
                        video_transforms2.uniform_crop(frames, self.crop[0], 1)[0],
                        video_transforms2.uniform_crop(frames, self.crop[0], 2)[0],
                        ])
            

        return frames

