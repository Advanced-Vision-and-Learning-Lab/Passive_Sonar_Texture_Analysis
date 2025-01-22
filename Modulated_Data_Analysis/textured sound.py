#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 17:14:37 2025

@author: jarin.ritu
"""

import numpy as np
import librosa
import soundfile as sf

def generate_textured_sound(duration, sampling_rate):
    """
    Generate a textured sound using a combination of noise and periodic patterns.
    """
    # White noise component
    white_noise = np.random.normal(0, 0.5, int(duration * sampling_rate))

    # Periodic pattern (e.g., sine wave modulating noise)
    time = np.linspace(0, duration, int(duration * sampling_rate))
    sine_wave = 0.5 * np.sin(2 * np.pi * 5 * time)  # Sine wave at 5 Hz
    modulated_noise = white_noise * (1 + sine_wave)

    # Random bursts (e.g., simulating rain or other textured sounds)
    burst_rate = 10  # Number of bursts per second
    burst_duration = 0.05  # Duration of each burst in seconds
    bursts = np.zeros_like(modulated_noise)
    for i in range(int(burst_rate * duration)):
        start = np.random.randint(0, len(modulated_noise) - int(burst_duration * sampling_rate))
        bursts[start : start + int(burst_duration * sampling_rate)] += np.random.normal(0.5, 0.3, int(burst_duration * sampling_rate))

    # Combine components
    textured_sound = modulated_noise + bursts

    # Normalize the sound to ensure it doesn't clip
    textured_sound = textured_sound / np.max(np.abs(textured_sound))

    return textured_sound

# Parameters
duration = 5  # Duration in seconds
sampling_rate = 44100  # Sampling rate in Hz
output_file = "textured_sound.wav"

# Generate and save the textured sound
textured_sound = generate_textured_sound(duration, sampling_rate)
sf.write(output_file, textured_sound, sampling_rate)

print(f"Textured sound saved as {output_file}")
