#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 17:56:57 2025
Updated with class-specific Gaussian transitions

@author: jarin.ritu
"""

import os
import numpy as np
import scipy.stats as stats
import soundfile as sf


# === Configuration ===
sampling_rate = 32000
duration = 5.0  # seconds
num_samples = int(sampling_rate * duration)
time = np.linspace(0, duration, num_samples, endpoint=False)
t_norm = np.linspace(0, 1, num_samples)  # normalized time
num_samples_per_class = 1

# === Frequency Bands for Each Class ===
freqs_list = {
    "Cargo": [2200, 3500, 5000, 6500],
    "Tanker": [3000, 4500, 6000, 7000],
    "Passengership": [9000, 10500, 12000, 13000],
    "Tug": [10000, 11500, 13000, 14000]
}


# === Amplitude Scaling Factors ===
amplitude_factors = {
    2200: 6.0, 3500: 4.5, 5000: 3.5, 6500: 2.5,   # Cargo
    3000: 5.0, 4500: 4.0, 6000: 3.0, 7000: 2.0,   # Tanker
    9000: 2.0, 10500: 1.2, 12000: 0.8, 13000: 0.6,  # Passengership
    10000: 1.8, 11500: 1.0, 13000: 0.6, 14000: 0.4  # Tug
}

# === Modulation Parameters ===
modulation_params = {
    "Cargo": {"depth": (0.3, 0.5), "rate": (5, 15)},
    "Tanker": {"depth": (0.4, 0.6), "rate": (3, 10)},
    "Passengership": {"depth": (0.2, 0.4), "rate": (10, 20)},
    "Tug": {"depth": (0.5, 0.7), "rate": (5, 25)}
}

# === Statistical Distribution Settings ===
rayleigh_scales = {"Cargo": 0.5, "Tanker": 0.3, "Passengership": 0.8, "Tug": 1.0}
k_shape = 14.0
k_scale = 1.0

# === Class-specific Gaussian transition parameters ===
gaussian_params = {
    "Cargo": (0.2, 0.1),            # Early Rayleigh → K
    "Tanker": (0.8, 0.1),           # Late K → Rayleigh
    "Passengership": (0.5, 0.3),    # Flat blend
    "Tug": (0.5, 0.15)              # Mid peak
}

# === Background Noise Generator ===
def generate_background_noise(num_samples, intensity=0.02):
    white = np.random.normal(0, intensity, num_samples)
    pink = np.random.normal(0, intensity / 2, num_samples) / np.sqrt(np.arange(1, num_samples + 1))
    return white + pink

# === Output Directory ===
output_dir = "mixed_texture"
os.makedirs(output_dir, exist_ok=True)
for cls in freqs_list.keys():
    os.makedirs(os.path.join(output_dir, cls), exist_ok=True)

dataset = {}

# === Signal Generation ===
for ship_class in freqs_list.keys():
    for sample_idx in range(num_samples_per_class):
        final_signal = np.zeros(num_samples)

        # Class-specific Gaussian transition
        mu, sigma = gaussian_params[ship_class]
        blend_weights = np.exp(-0.5 * ((t_norm - mu) / sigma) ** 2)
        blend_weights /= np.max(blend_weights)

        for freq in freqs_list[ship_class]:
            rayleigh_amps = stats.rayleigh.rvs(scale=rayleigh_scales[ship_class], size=num_samples)
            k_amps = stats.gamma.rvs(k_shape, scale=k_scale, size=num_samples)

            rayleigh_amps /= np.max(rayleigh_amps)
            k_amps /= np.max(k_amps)

            amplitudes = blend_weights * rayleigh_amps + (1 - blend_weights) * k_amps

            depth_range = modulation_params[ship_class]["depth"]
            rate_range = modulation_params[ship_class]["rate"]
            mod_depth = np.random.uniform(*depth_range)
            mod_rate = np.random.uniform(*rate_range)
            modulation = 1 + mod_depth * np.sin(2 * np.pi * mod_rate * time)

            sine_wave = amplitude_factors[freq] * amplitudes * np.sin(2 * np.pi * freq * time) * modulation
            final_signal += sine_wave

        final_signal += generate_background_noise(num_samples, intensity=0.05)
        final_signal /= np.max(np.abs(final_signal))

        label = f"{ship_class}_{sample_idx}"
        dataset[label] = final_signal
        path = os.path.join(output_dir, ship_class, f"{label}.wav")
        sf.write(path, final_signal, sampling_rate)
        print(f"✅ Saved: {path}")

