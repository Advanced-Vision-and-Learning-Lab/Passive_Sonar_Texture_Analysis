#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 12:11:18 2024

@author: jarin.ritu
"""

import torch
import matplotlib.pyplot as plt

# Load a generated spectrogram
generated_sample = torch.load("DeepShip_Spectrograms/Cargo/20171111g-11/11_Cargo-Segment_6.pt")

# Visualize the spectrogram
plt.imshow(generated_sample.squeeze(), cmap="viridis")
plt.colorbar()
plt.title("Original Spectrogram")
plt.show()

# Load a generated spectrogram


spectrogram = torch.load("Generated_Spectrograms/generated_class_0_0.pt")
plt.imshow(spectrogram.numpy(), aspect='auto', origin='lower', cmap='viridis')
plt.colorbar()
plt.title("Generated Spectrogram")
plt.show()
