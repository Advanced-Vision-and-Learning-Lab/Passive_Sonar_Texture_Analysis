#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthetic Ship Sounds with High Statistical Texture 

@author: jarin.ritu
"""

import os
import numpy as np
import soundfile as sf
from scipy.stats import rayleigh, gamma

# ✅ Constants
sampling_rate = 32000
duration = 5.0
num_samples = int(sampling_rate * duration)
time = np.linspace(0, duration, num_samples, endpoint=False)
num_samples_per_class = 10000

# ✅ Statistical parameters per class
statistical_params = {
    "Cargo": {"Rayleigh": 0.5, "K": (12, 1.0)},
    "Tanker": {"Rayleigh": 0.6, "K": (14, 1.2)},
    "Passenger": {"Rayleigh": 0.8, "K": (10, 1.5)},
    "Tug": {"Rayleigh": 0.7, "K": (8, 1.8)}
}

# ✅ Background noise generator
def generate_background_noise(num_samples, intensity=0.02):
    return np.random.normal(0, intensity, num_samples)

# ✅ Create output directory
output_dir = "stat_vis"
os.makedirs(output_dir, exist_ok=True)
for ship_class in statistical_params.keys():
    os.makedirs(os.path.join(output_dir, ship_class), exist_ok=True)

# ✅ Dataset container
dataset = {}

# ✅ Signal generation per class
for ship_class, params in statistical_params.items():
    for sample_idx in range(num_samples_per_class):
        # ✅ Choose distribution
        use_rayleigh = np.random.rand() > 0.5

        if use_rayleigh:
            scale = params["Rayleigh"]
            amplitudes = rayleigh.rvs(scale=scale, size=num_samples)
        else:
            shape, scale = params["K"]
            amplitudes = gamma.rvs(shape, scale=scale, size=num_samples)

        amplitudes /= np.max(amplitudes)

        # ✅ Amplitude envelope for modulation
        envelope = np.linspace(0.5, 1.0, num_samples) * np.random.uniform(0.7, 1.3)
        amplitudes *= envelope

        # ✅ Random frequency and phase
        base_freq = np.random.normal(loc=4000, scale=500) + np.random.uniform(-1000, 1000)
        phase_offset = np.random.uniform(0, 2 * np.pi)

        # ✅ Generate modulated signal
        signal = amplitudes * np.sin(2 * np.pi * base_freq * time + phase_offset)

        # ✅ Add low-frequency machinery hum
        low_freq = np.random.uniform(50, 300)
        low_hum = 0.2 * np.sin(2 * np.pi * low_freq * time)
        signal += low_hum

        # ✅ Add background noise
        noise_intensity = np.random.uniform(0.005, 0.02)
        signal += generate_background_noise(num_samples, intensity=noise_intensity)

        # ✅ Random dropout (occlusion-like effect)
        if np.random.rand() < 0.3:
            dropout_start = np.random.randint(0, num_samples - 500)
            signal[dropout_start:dropout_start + 500] *= np.random.uniform(0.0, 0.3)

        # ✅ Normalize signal
        signal /= np.max(np.abs(signal))

        # ✅ Save
        label = f"{ship_class}_{sample_idx}"
        dataset[label] = signal
        file_path = os.path.join(output_dir, ship_class, f"{label}.wav")
        sf.write(file_path, signal, sampling_rate)
        print(f"✅ Saved: {file_path}")