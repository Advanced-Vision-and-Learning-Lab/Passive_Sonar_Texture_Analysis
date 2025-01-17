#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 10:27:58 2025

@author: jarin.ritu
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchaudio
from nnAudio import features
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
import numpy as np
import pdb
import random
from librosa.util.exceptions import ParameterError
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import numpy as np
import torch.nn as nn
import torchaudio
from EDM import EDM
import matplotlib.pyplot as plt

class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1
        
    def forward(self, x):
        return torch.transpose(x, self.dim0, self.dim1)

    
class Feature_Extraction_Layer(nn.Module):
    def __init__(self, input_feature, window_length, window_size, hop_size, mel_bins, fmin, fmax, classes_num,
                 hop_length, sample_rate=8000, RGB=False, downsampling_factor=2, frame_shift=10.0):
        super(Feature_Extraction_Layer, self).__init__()

        # Convert window and hop length to ms
        window_length /= 1000
        hop_length /= 1000
        window = 'hann'
        center = False
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        self.frame_shift = frame_shift  
        self.bn = nn.BatchNorm2d(1024)
    

        if RGB:
            num_channels = 3
            MFCC_padding = nn.ZeroPad2d((3, 6, 16, 16))
        else:
            num_channels = 1
            MFCC_padding = nn.ZeroPad2d((1, 0, 4, 0))
        
        
        self.num_channels = num_channels
        self.input_feature = input_feature

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



       # Return Mel Spectrogram that is 48 x 48  MITLL new
        self.Mel_Spectrogram = nn.Sequential(
        features.mel.MelSpectrogram(
            sample_rate,
            n_mels=1024,  # 128 Mel bins
            win_length=8192,  # Window length of 8192
            hop_length=780,  # Hop length of 1024
            n_fft=8192,  # Set n_fft equal to win_length for STFT
            verbose=False,
            fmax=sample_rate / 4 
        ),
        nn.ZeroPad2d((1, 4, 0, 4))  # Optional padding
    )


        
        # Initialize new FBankLayer

        self.features = {'Log_Mel_Spectrogram':self.Log_Mel_Spectrogram,'Mel_Spectrogram':self.Mel_Spectrogram}


    
    def forward(self, x):
        x = self.features[self.input_feature](x)
        # pdb.set_trace()
        x = x.repeat(1, self.num_channels, 1, 1)

        return x
    
def adjust_feature_params(n_mels, hop_length, window_size, max_level):
    # Ensure n_mels is divisible by 2^(max_level - 1)
    divisor = 2 ** (max_level - 1)
    n_mels = (n_mels // divisor) * divisor

    # Ensure time steps (from hop_length) are compatible
    # Assuming a sample length, calculate an appropriate hop length
    hop_length = (hop_length // divisor) * divisor
    window_size = max(window_size, hop_length * 2)  # Ensure window is large enough

    return n_mels, hop_length, window_size
def pad_signal(signal, hop_size, window_size, max_level):
    """
    Pad the signal to ensure the resulting time steps are divisible by 2^(max_level - 1).
    """
    # Calculate the divisor
    divisor = 2 ** (max_level - 1)

    # Calculate the number of time steps
    signal_length = signal.shape[-1]
    time_steps = (signal_length - window_size) // hop_size + 1

    # Ensure time steps are divisible
    extra_steps = (divisor - (time_steps % divisor)) % divisor
    extra_samples = extra_steps * hop_size
    padding = (0, extra_samples)  # Pad only at the end

    # Pad the signal
    return F.pad(signal, padding, mode="constant", value=0)

def normalize_response(response):
    min_val = response.min()
    max_val = response.max()
    return (response - min_val) / (max_val - min_val)

if __name__ == "__main__":
    # Step 1: Load the audio signal
    audio_path = "5_Cargo-Segment_1.wav"  # Replace with your audio file path
    waveform, sample_rate = torchaudio.load(audio_path)


### 155: 1000x1024; 1024:157x1024
    # Step 2: Ensure the waveform has the correct shape
    if waveform.ndim == 1:  # If mono, add channel dimension
        waveform = waveform.unsqueeze(0)
    # Parameters
    max_level = 3  # Laplacian pyramid levels
    original_n_mels = 1024
    original_hop_length = 1024
    original_window_size = 8192
    # original_n_mels = 101
    # original_hop_length = 320
    # original_window_size = 1024
    # Adjust feature parameters
    n_mels, hop_size, window_size = adjust_feature_params(
        original_n_mels, original_hop_length, original_window_size, max_level
    )

    # Print adjusted values for verification
    print(f"Adjusted n_mels: {n_mels}, hop_size: {hop_size}, window_size: {window_size}")

    #1sec = 312
    # Step 3: Extract the spectrogram using the desired feature type
    feature_extractor = Feature_Extraction_Layer(
        input_feature="Log_Mel_Spectrogram",  # Specify the desired feature type
        window_length=83,
        window_size=window_size,
        hop_size=hop_size,
        mel_bins=n_mels,
        fmin=50,
        fmax=14000,
        classes_num=527,
        hop_length=21,
        sample_rate=sample_rate,
        RGB=False
    )
    # Pad the waveform to ensure compatibility
    waveform = pad_signal(waveform, hop_size, window_size, max_level)
    
    # Verify the padded signal length
    print(f"Padded signal length: {waveform.shape[-1]}")
    spectrogram = feature_extractor(waveform)

    # Step 4: Pass the spectrogram to the EDM module
    in_channels = spectrogram.shape[1]  # Number of channels from spectrogram
    edm = EDM(in_channels=in_channels, max_level=max_level, fusion='all')
    edge_response = edm(spectrogram)
    # pdb.set_trace()


    # Aggregate responses by taking the max across the 8 channels
    aggregated_responses = [response[0].max(dim=0)[0].cpu().detach().numpy() for response in edge_response]
    
    # Plot the aggregated responses (max across channels)
    fig, axes = plt.subplots(1, len(aggregated_responses), figsize=(20, 5))
    for i, (agg_response, ax) in enumerate(zip(aggregated_responses, axes)):
        # Normalize the response for better visualization
        agg_response = (agg_response - agg_response.min()) / (agg_response.max() - agg_response.min())
    
        # Plot the aggregated response
        im = ax.imshow(agg_response, cmap='coolwarm', aspect='auto')
        ax.set_title(f'Edge Detected: Level {i}', fontsize=16)
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()
    pdb.set_trace()
    # Print shapes for verification
    print("Spectrogram shape:", spectrogram.shape)
    print("EDM output shape:", edge_response[0].shape)
    
    
## plot all 24
# Plot all 8 channels for each level
# for level_idx, response in enumerate(edge_response):
#     fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # 2x4 grid for 8 channels
#     axes = axes.flatten()  # Flatten for easy indexing
#     for channel_idx, ax in enumerate(axes):
#         # Normalize and extract the channel
#         response_np = normalize_response(response[0, channel_idx].cpu().detach().numpy())
#         im = ax.imshow(response_np, cmap='coolwarm', aspect='auto', vmin=-0.2, vmax=0.65)
#         ax.set_title(f'Level {level_idx}, Channel {channel_idx}')
#         plt.colorbar(im, ax=ax)
#     plt.tight_layout()
#     plt.show()
