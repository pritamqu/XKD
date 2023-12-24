# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 17:15:56 2021

@author: pritam
"""

# calculate norm stats based on kinetics-sound

import torch
import numpy as np
from datasets.loader_1.kinetics_sound_new import KineticsSound
from tools import my_paths
from datasets.audiotransforms.audio_transforms import AudioPrep
from datasets.audiotransforms.to_spec import MelSpectrogramLibrosa
import matplotlib.pylab as plt

sample_rate=16000
n_fft=1024
hop_length=160
n_mels=80
f_min=0
f_max=None
duration=2

def plot_spectrogram(spec, title=None, ylabel='mel', aspect='auto', xmax=None):
    from matplotlib import pyplot as plt
    import librosa
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or 'Spectrogram (db)')
    axs.set_ylabel(ylabel)
    axs.set_xlabel('frame')
    # im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
    im = axs.imshow(spec, origin='lower', aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.show(block=False)

db = KineticsSound(
        DATA_PATH = my_paths('local', 'kinetics_sound')[-1],
              subset = 'train',
              return_video=False,
              return_audio=True,
              audio_clip_duration=duration,
              audio_fps=sample_rate,
              audio_fps_out=None,
              audio_transform=None,
              return_labels=False,
              return_index=False,
              max_offsync_augm=0,
              mode='clip',
              submode=None,
              clips_per_video=1)

basic_prep = AudioPrep(duration=duration, trim_pad=True, to_tensor=True)
to_melspecgram = MelSpectrogramLibrosa(
            fs=sample_rate,
            n_fft=n_fft,
            shift=hop_length,
            n_mels=n_mels,
            fmin=f_min,
            fmax=f_max,
        ) 


stack = []
for i in range(len(db)):
    wav = db.__getitem__(i)['audio']
    
    wav = basic_prep(wav)
    wav = wav[0]
    lms = (to_melspecgram(wav) + torch.finfo().eps).log().unsqueeze(0)
    # plot_spectrogram(lms.numpy().transpose(1, 2, 0),)
    stack.append(lms)
    print(i)
    
X = np.hstack(stack)
norm_stats = np.array([X.mean(), X.std()])
# array([-5.3688107,  4.410007 ], dtype=float32)




################ rough ###############

# from datasets.audiotransforms.spec_aug import TimeWarp, MaskAlongAxis

# fmask = MaskAlongAxis(mask_width_range=(0, 10), num_mask=2, dim='freq', replace_with_zero=False)
# tmask = MaskAlongAxis(mask_width_range=(0, 20), num_mask=2, dim='time', replace_with_zero=False)
# timewrap = TimeWarp(window=80)

# i=1234
# wav = db.__getitem__(i)['audio']
# wav = basic_prep(wav)
# wav = wav[0]

# ########
# lms = (to_melspecgram(wav) + torch.finfo().eps).log().unsqueeze(0) # (Batch, Freq, Length)
# lms = (lms-5.3688107)/4.410007
# plot_spectrogram(lms.numpy().transpose(1, 2, 0),) 

# outf = fmask(lms.transpose(2, 1))[0].transpose(2, 1) # takes input as spec: (Batch, Length, Freq)
# plot_spectrogram(outf.numpy().transpose(1, 2, 0),)

# outt = tmask(lms.transpose(2, 1))[0].transpose(2, 1)
# plot_spectrogram(outt.numpy().transpose(1, 2, 0),)

# outtw = timewrap(lms.transpose(2, 1))[0].transpose(2, 1)
# plot_spectrogram(outtw.numpy().transpose(1, 2, 0),)


