#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 11:31:24 2024

@author: jarin.ritu
"""
import torchaudio
import torch
import os
from torchvision.transforms import Resize

def trim_or_pad_signal(waveform, target_length, sample_rate):
    target_samples = int(target_length * sample_rate)
    if waveform.size(1) > target_samples:  # Trim
        waveform = waveform[:, :target_samples]
    elif waveform.size(1) < target_samples:  # Pad
        pad_size = target_samples - waveform.size(1)
        waveform = torch.nn.functional.pad(waveform, (0, pad_size))
    return waveform

def extract_spectrograms_nested(audio_path, save_dir, target_classes, sample_rate=16000, n_mels=48, win_length=250, hop_length=64, target_length=5):
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=250,
        hop_length=64,
        win_length=250,
        n_mels=126  # Matches n_fft / 2 + 1
    )


    # Add resize transform for 128x128
    resize_transform = Resize((128, 128))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for class_name in target_classes:
        class_path = os.path.join(audio_path, class_name)
        class_save_dir = os.path.join(save_dir, class_name)
        if not os.path.exists(class_save_dir):
            os.makedirs(class_save_dir)
        
        # Traverse subdirectories
        for root, _, files in os.walk(class_path):
            for file in files:
                if file.endswith('.wav'):
                    file_path = os.path.join(root, file)
                    waveform, sr = torchaudio.load(file_path)
                    waveform = trim_or_pad_signal(waveform, target_length=target_length, sample_rate=sample_rate)
                    
                    # Generate Mel Spectrogram
                    mel_spec = transform(waveform)

                    # Resize spectrogram to 128x128
                    mel_spec_resized = resize_transform(mel_spec.unsqueeze(0)).squeeze(0)

                    # Save with original subdirectory structure
                    rel_path = os.path.relpath(root, class_path)
                    save_subdir = os.path.join(class_save_dir, rel_path)
                    if not os.path.exists(save_subdir):
                        os.makedirs(save_subdir)
                    
                    save_path = os.path.join(save_subdir, file.replace('.wav', '.pt'))
                    torch.save(mel_spec_resized, save_path)
                    print(f"Saved: {file_path} as {save_path}")

# Example usage
extract_spectrograms_nested(
    audio_path="DeepShip",
    save_dir="DeepShip_Spectrograms",
    target_classes=["Cargo"],
    sample_rate=16000,
    target_length=5  # Duration of 5 seconds
)
