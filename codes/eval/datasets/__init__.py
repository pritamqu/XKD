import os
from datasets.loader.audioset import AudioSet
from datasets.loader.kinetics import Kinetics
from datasets.loader.kinetics_sound import KineticsSound
from datasets.loader.ucf import UCF
from datasets.loader.hmdb import HMDB
from datasets.loader.esc import ESC
from datasets.loader.dcase import DCASE
from datasets.loader.vggsound import VGGSound
from datasets.loader.fsd50 import FSD
# from datasets.loader.fsd50_hdf5 import FSD
# from datasets.loader.esc_hdf5 import ESC -> it behaves crazy
from datasets.loader.ssv2 import SSV2
from datasets.loader.openmic import OpenMic
from datasets.loader.charades import Charades

import random
import torch


def get_dataset(root, dataset_kwargs, video_transform=None, audio_transform=None, split='train'):
    name = dataset_kwargs['name']
          
    ## action recognition
    if name=='ucf101':
        return UCF(
            DATA_PATH = os.path.join(root, 'UCF-101'),
                 ANNO_PATH = os.path.join(root, 'ucfTrainTestlist'),
                 subset = dataset_kwargs[split]['split'].format(fold=dataset_kwargs['fold']),
                 return_video=True,
                 video_clip_duration=dataset_kwargs['clip_duration'],
                 video_fps=dataset_kwargs['video_fps'],
                 video_transform=video_transform,
                 return_audio=False,
                 return_labels=True,
                 return_index=True,
                 mode=dataset_kwargs[split]['mode'],
                 clips_per_video=dataset_kwargs[split]['clips_per_video'],)

    elif name=='hmdb51':
        return HMDB(
            DATA_PATH = os.path.join(root, 'HMDB-51'),
                 ANNO_PATH = os.path.join(root, 'testTrainMulti_7030_splits'),
                 subset = dataset_kwargs[split]['split'].format(fold=dataset_kwargs['fold']),
                 return_video=True,
                 video_clip_duration=dataset_kwargs['clip_duration'],
                 video_fps=dataset_kwargs['video_fps'],
                 video_transform=video_transform,
                 return_audio=False,
                 return_labels=True,
                 return_index=True,
                 mode=dataset_kwargs[split]['mode'],
                 clips_per_video=dataset_kwargs[split]['clips_per_video'],)
    
    elif name=='ssv2':
        return SSV2(
            DATA_PATH = root,
                 subset = dataset_kwargs[split]['split'],
                 return_video=True,
                 video_clip_duration=dataset_kwargs['clip_duration'],
                 video_fps=dataset_kwargs['video_fps'],
                 video_transform=video_transform,
                 return_labels=True,
                 return_index=True,
                 mode=dataset_kwargs[split]['mode'],
                 clips_per_video=dataset_kwargs[split]['clips_per_video'],)
    
    elif name=='charades':
        return Charades(
            root = os.path.join(root, 'Charades_v1_rgb'),
                 anno = root,
                 split = dataset_kwargs[split]['split'],
                 video_clip_duration=dataset_kwargs['clip_duration'],
                 video_fps=dataset_kwargs['video_fps'],
                 video_transform=video_transform,
                 return_labels=True,
                 return_index=True,
                 mode=dataset_kwargs[split]['mode'],
                 clips_per_video=dataset_kwargs[split]['clips_per_video'],)    
    
    ## sound classification        
    elif name=='esc50':
        return ESC(
            DATA_PATH = os.path.join(root, 'audio'),
                 ANNO_PATH = os.path.join(root, 'meta'),
                 subset = dataset_kwargs[split]['split'].format(fold=dataset_kwargs['fold']),
                 audio_clip_duration=dataset_kwargs['audio_clip_duration'],
                 audio_fps=dataset_kwargs['audio_fps'],
                 audio_fps_out=dataset_kwargs['audio_fps_out'],
                 audio_transform=audio_transform,
                 return_labels=True,
                 return_index=True,
                 mode=dataset_kwargs[split]['mode'],
                 clips_per_audio=dataset_kwargs[split]['clips_per_video'],
            )
        # return ESC(
        #     DATA_PATH = root,
        #          # ANNO_PATH = os.path.join(root, 'meta'),
        #          subset = dataset_kwargs[split]['split'].format(fold=dataset_kwargs['fold']),
        #          audio_clip_duration=dataset_kwargs['audio_clip_duration'],
        #          audio_fps=dataset_kwargs['audio_fps'],
        #          audio_fps_out=dataset_kwargs['audio_fps_out'],
        #          audio_transform=audio_transform,
        #          return_labels=True,
        #          return_index=True,
        #          mode=dataset_kwargs[split]['mode'],
        #          clips_per_audio=dataset_kwargs[split]['clips_per_video'],
        #     )

    elif name=='dcase':
        return DCASE(
            DATA_PATH = os.path.join(root),
                 subset = dataset_kwargs[split]['split'],
                 audio_clip_duration=dataset_kwargs['audio_clip_duration'],
                 audio_fps=dataset_kwargs['audio_fps'],
                 audio_fps_out=dataset_kwargs['audio_fps_out'],
                 audio_transform=audio_transform,
                 return_labels=True,
                 return_index=True,
                 mode=dataset_kwargs[split]['mode'],
                 clips_per_audio=dataset_kwargs[split]['clips_per_video'],
            )

    elif name=='vggsound':
        return VGGSound(
            DATA_PATH = os.path.join(root),
                 subset = dataset_kwargs[split]['split'],
                 audio_clip_duration=dataset_kwargs['audio_clip_duration'],
                 audio_fps=dataset_kwargs['audio_fps'],
                 audio_fps_out=dataset_kwargs['audio_fps_out'],
                 audio_transform=audio_transform,
                 return_labels=True,
                 return_index=True,
                 mode=dataset_kwargs[split]['mode'],
                 clips_per_audio=dataset_kwargs[split]['clips_per_video'],
            )
    
    elif name=='fsd50':
        return FSD(
            ROOT = os.path.join(root),
            subset = dataset_kwargs[split]['split'],
            audio_clip_duration=dataset_kwargs['audio_clip_duration'],
            audio_fps=dataset_kwargs['audio_fps'],
            audio_fps_out=dataset_kwargs['audio_fps_out'],
            audio_transform=audio_transform,
            return_labels=True,
            return_index=True,
            mode=dataset_kwargs[split]['mode'],
            clips_per_audio=dataset_kwargs[split]['clips_per_video'],
            )

    elif name=='openmic':
        return OpenMic(
            ROOT = os.path.join(root),
            subset = dataset_kwargs[split]['split'],
            audio_clip_duration=dataset_kwargs['audio_clip_duration'],
            audio_fps=dataset_kwargs['audio_fps'],
            audio_fps_out=dataset_kwargs['audio_fps_out'],
            audio_transform=audio_transform,
            return_labels=True,
            return_index=True,
            mode=dataset_kwargs[split]['mode'],
            clips_per_audio=dataset_kwargs[split]['clips_per_video'],
            )

    elif name=='kinetics400':
        return Kinetics(
            DATA_PATH = os.path.join(root),
                  subset = dataset_kwargs[split]['split'],
                  # return_video=dataset_kwargs['return_video'],
                  video_clip_duration=dataset_kwargs['clip_duration'],
                  video_fps=dataset_kwargs['video_fps'],
                  video_transform=video_transform,
                  # return_audio=dataset_kwargs['return_audio'],
                  # audio_clip_duration=dataset_kwargs['audio_clip_duration'],
                  # audio_fps=dataset_kwargs['audio_fps'],
                  # audio_fps_out=dataset_kwargs['audio_fps_out'],
                  # audio_transform=audio_transform,
                  return_labels=True,
                  return_index=True,
                  mode=dataset_kwargs[split]['mode'],
                  clips_per_video=dataset_kwargs[split]['clips_per_video'],)

    elif name=='kinetics_sound':
        return KineticsSound(
            DATA_PATH = os.path.join(root),
                 subset = dataset_kwargs[split]['split'],
                 return_video=dataset_kwargs['return_video'],
                 video_clip_duration=dataset_kwargs['clip_duration'] if dataset_kwargs['return_video'] else None,
                 video_fps=dataset_kwargs['video_fps'] if dataset_kwargs['return_video'] else None,
                 video_transform=video_transform if dataset_kwargs['return_video'] else None,
                 return_audio=dataset_kwargs['return_audio'],
                 audio_clip_duration=dataset_kwargs['audio_clip_duration'] if dataset_kwargs['return_audio'] else None,
                 audio_fps=dataset_kwargs['audio_fps'] if dataset_kwargs['return_audio'] else None,
                 audio_fps_out=dataset_kwargs['audio_fps_out'] if dataset_kwargs['return_audio'] else None,
                 audio_transform=audio_transform if dataset_kwargs['return_audio'] else None,
                 return_labels=True,
                 return_index=True,
                 mode=dataset_kwargs[split]['mode'],
                 clips_per_video=dataset_kwargs[split]['clips_per_video'],)    

    elif name=='audioset':
        return AudioSet(
            DATA_PATH = os.path.join(root),
                  subset = dataset_kwargs[split]['split'],
                  return_video=dataset_kwargs['return_video'],
                  video_clip_duration=dataset_kwargs['clip_duration'],
                  video_fps=dataset_kwargs['video_fps'],
                  video_transform=video_transform,
                  return_audio=dataset_kwargs['return_audio'],
                  audio_clip_duration=dataset_kwargs['audio_clip_duration'],
                  audio_fps=dataset_kwargs['audio_fps'],
                  audio_fps_out=dataset_kwargs['audio_fps_out'],
                  audio_transform=audio_transform,
                  return_labels=False,
                  return_index=False,
                  max_offsync_augm=0,
                  mode=dataset_kwargs[split]['mode'],
                  submode=dataset_kwargs[split]['submode'],
                  clips_per_video=dataset_kwargs[split]['clips_per_video'],)
    
    else:
        raise NotImplementedError(f'{name} is not available.')
        

def fetch_subset(dataset, size=None):
    if size is None:
        size = len(dataset.classes)
    indices = random.sample(range(len(dataset)), size)
    samples = torch.utils.data.Subset(dataset, indices=indices)
    # samples = subset(dataset, indices=indices)
    return samples

class FetchSubset(torch.utils.data.Subset):

    def __init__(self, dataset, size=None):
        self.dataset = dataset
        if size is None:
            size = len(dataset.classes)
        self.indices = random.sample(range(len(dataset)), size)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.dataset, name)
    