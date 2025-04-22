#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthetic Ship Sounds with High Structural Texture 

@author: jarin.ritu
"""

import os
import numpy as np
import soundfile as sf

# ✅ Constants
sampling_rate = 32000  # 32 kHz
duration = 5.0  # 5 seconds per sample
num_samples = int(sampling_rate * duration)
time = np.linspace(0, duration, num_samples, endpoint=False)
num_samples_per_class = 10000
segments = 5
segment_samples = num_samples // segments

# ✅ Class-specific configuration for structural texture
structural_params = {
    "Cargo": {"f_base": 2800, "envelope": "triangular", "noise_std": 0.010},
    "Tanker": {"f_base": 3600, "envelope": "exponential", "noise_std": 0.004},
    "Passenger": {"f_base": 5400, "envelope": "ramp-up", "noise_std": 0.006},
    "Tug": {"f_base": 6200, "envelope": "ramp-down", "noise_std": 0.002}
}

def generate_envelope(shape, t):
    if shape == "triangular":
        return 1 - 2 * np.abs(t - 0.5)
    elif shape == "exponential":
        return np.exp(-5 * t)
    elif shape == "ramp-up":
        return t
    elif shape == "ramp-down":
        return 1 - t
    else:
        return np.ones_like(t)

output_dir = "synthetic_structural_texture"
os.makedirs(output_dir, exist_ok=True)
for ship_class in structural_params:
    os.makedirs(os.path.join(output_dir, ship_class), exist_ok=True)

dataset = {}
for ship_class, params in structural_params.items():
    for sample_idx in range(num_samples_per_class):
        f_base = params["f_base"]
        envelope_shape = params["envelope"]
        noise_std = params["noise_std"]

        # Normalize time to [0,1] for envelope
        norm_time = (time - time.min()) / (time.max() - time.min())
        envelope = generate_envelope(envelope_shape, norm_time)

        # 3 harmonics with jitter
        # 3 harmonics with jitter
        signal = np.zeros(num_samples)
        for k in range(1, 4):
            jitter = np.random.uniform(-10, 10)  # Smaller jitter for clearer structure
            harmonic_freq = k * f_base + jitter
            signal += (1.0 / k) * np.sin(2 * np.pi * harmonic_freq * time)


        signal *= envelope
        signal += np.random.normal(0, noise_std, size=num_samples)
        signal /= np.max(np.abs(signal) + 1e-6)

        dataset[f"{ship_class}_{sample_idx}"] = signal
        file_path = os.path.join(output_dir, ship_class, f"{ship_class}_{sample_idx}.wav")
        sf.write(file_path, signal, sampling_rate)
        if sample_idx == 0:
            print(f"✅ Saved example: {file_path}")

