#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 18:45:34 2025

@author: jarin.ritu
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp
import soundfile as sf

def generate_textured_audio(duration, sample_rate):
    """
    Generate textured audio with varying frequencies and amplitudes.
    Example: White noise + Amplitude modulation + Chirp signals
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # White noise
    noise = np.random.normal(0, 0.5, t.shape)
    
    # Amplitude modulation
    modulation = 0.5 * (1 + np.sin(2 * np.pi * 0.25 * t))  # Low-frequency modulator
    
    # Chirp signal
    chirp_signal = chirp(t, f0=50, f1=sample_rate // 2, t1=duration, method='linear')
    
    # Combine signals
    textured_audio = noise + modulation * chirp_signal
    return textured_audio / np.max(np.abs(textured_audio))  # Normalize

def generate_non_textured_audio(duration, sample_rate):
    """
    Generate non-textured audio with simple, repeating patterns.
    Example: Pure sine wave
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    sine_wave = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave (A4)
    return sine_wave

# Parameters
sample_rate = 16000  # 16 kHz sampling rate
duration = 5  # 5 seconds

# Generate textured and non-textured audio
textured_audio = generate_textured_audio(duration, sample_rate)
non_textured_audio = generate_non_textured_audio(duration, sample_rate)

# Save the audio to files
sf.write("textured_audio.wav", textured_audio, sample_rate)
sf.write("non_textured_audio.wav", non_textured_audio, sample_rate)

# Plot the waveforms
plt.figure(figsize=(10, 4))
plt.plot(textured_audio[:sample_rate])
plt.title("Textured Audio (1 second)")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(non_textured_audio[:sample_rate])
plt.title("Non-Textured Audio (1 second)")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.show()
