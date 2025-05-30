#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 09:51:01 2024

@author: jarin.ritu
"""

import torch.nn as nn
import torch
from torchlibrosa.stft import Spectrogram, LogmelFilterBank


class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1
        
    def forward(self, x):
        return torch.transpose(x, self.dim0, self.dim1)
class Feature_Extraction_Layer(nn.Module):
    def __init__(self, input_feature, window_length,window_size, hop_size, mel_bins, fmin, fmax, classes_num,
                 hop_length, sample_rate=8000, RGB=False,RGB_Teacher=False,downsampling_factor=2):
        super(Feature_Extraction_Layer, self).__init__()

        # Convert window and hop length to ms
        window_length /= 1000
        hop_length /= 1000
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None        
        self.num_channels = 1
        self.input_feature = input_feature
        self.bn = nn.BatchNorm2d(64)

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # Spec augmenter
        # self.spec_augmenter = SpecAugmentation(time_drop_width=48, time_stripes_num=2, 
        #     freq_drop_width=8, freq_stripes_num=2)

        # Return Mel Spectrogram that is 48 x 48
        self.Log_Mel_Spectrogram = nn.Sequential(self.spectrogram_extractor,
                                                self.logmel_extractor,
                                                Transpose(1, 3),
                                                self.bn,
                                                Transpose(1, 3))
        

        self.features = {'Log_Mel_Spectrogram':self.Log_Mel_Spectrogram}

    def forward(self, x):
        x = x.squeeze(1)
        #Extract audio feature
        x = self.features[self.input_feature](x)
        
        #Repeat channel dimension if needed (CNNs)
        x = x.repeat(1, self.num_channels,1,1)
        
        return x
