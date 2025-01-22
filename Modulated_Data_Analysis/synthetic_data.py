#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 08:41:46 2025

@author: jarin.ritu
"""

import numpy as np
from scipy.stats import rayleigh, kappa4
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
import os

# Directory to save datasets
SAVE_DIR = "./synthetic_dataset2"
os.makedirs(SAVE_DIR, exist_ok=True)

# Generate Rayleigh Distribution Signals
def generate_rayleigh_signals(num_samples, scale, signal_length):
    return [np.random.rayleigh(scale=scale, size=signal_length) for _ in range(num_samples)]

# Generate K-Distribution Signals
def generate_k_signals(num_samples, h, k, signal_length):
    """
    Generate synthetic signals based on the K-distribution.

    Parameters:
    - num_samples: Number of signals to generate.
    - h: Shape parameter h of the K-distribution.
    - k: Shape parameter k of the K-distribution.
    - signal_length: Length of each signal.

    Returns:
    - List of synthetic signals.
    """
    return [kappa4.rvs(h=h, k=k, size=signal_length) for _ in range(num_samples)]

# Generate Periodic Signals
def generate_periodic_signals(num_samples, freq, signal_length, sample_rate):
    t = np.linspace(0, signal_length / sample_rate, signal_length)
    return [np.sin(2 * np.pi * freq * t) for _ in range(num_samples)]

# Combine Periodic and Aperiodic Signals
def generate_mixed_signals(periodic_signals, noise_scale):
    return [signal + np.random.normal(0, noise_scale, len(signal)) for signal in periodic_signals]

# Convert 1D Signals to Spectrograms
def signals_to_spectrograms(signals, fs, nperseg):
    spectrograms = []
    for signal in signals:
        f, t, Sxx = spectrogram(signal, fs=fs, nperseg=nperseg)
        spectrograms.append(Sxx)
    return spectrograms

# Save Spectrograms as Images
def save_spectrograms(spectrograms, labels, save_dir):
    for idx, (Sxx, label) in enumerate(zip(spectrograms, labels)):
        plt.figure(figsize=(6, 4))
        plt.pcolormesh(np.log(Sxx + 1e-8), cmap='viridis')
        plt.axis('off')
        plt.savefig(f"{save_dir}/{label}_{idx}.png", bbox_inches='tight', pad_inches=0)
        plt.close()

# Parameters
NUM_SAMPLES = 10
SIGNAL_LENGTH = 4800
SAMPLE_RATE = 1600
NPERSEG = 256

# Generate Datasets
rayleigh_signals = generate_rayleigh_signals(NUM_SAMPLES, scale=2.0, signal_length=SIGNAL_LENGTH)
k_signals = generate_k_signals(NUM_SAMPLES, h=2.0, k=0.5, signal_length=SIGNAL_LENGTH)


periodic_signals = generate_periodic_signals(NUM_SAMPLES, freq=5, signal_length=SIGNAL_LENGTH, sample_rate=SAMPLE_RATE)
mixed_signals = generate_mixed_signals(periodic_signals, noise_scale=0.5)
# Convert periodic signals into spectrograms
periodic_spectrograms = signals_to_spectrograms(periodic_signals, fs=SAMPLE_RATE, nperseg=NPERSEG)

# Save spectrograms for periodic signals
save_spectrograms(periodic_spectrograms, labels=["Periodic"] * NUM_SAMPLES, save_dir=SAVE_DIR)



# Convert to Spectrograms
rayleigh_spectrograms = signals_to_spectrograms(rayleigh_signals, fs=SAMPLE_RATE, nperseg=NPERSEG)
k_spectrograms = signals_to_spectrograms(k_signals, fs=SAMPLE_RATE, nperseg=NPERSEG)
mixed_spectrograms = signals_to_spectrograms(mixed_signals, fs=SAMPLE_RATE, nperseg=NPERSEG)


# Save Spectrograms
save_spectrograms(rayleigh_spectrograms, labels=["Rayleigh"] * NUM_SAMPLES, save_dir=SAVE_DIR)
save_spectrograms(k_spectrograms, labels=["K-Distribution"] * NUM_SAMPLES, save_dir=SAVE_DIR)
save_spectrograms(mixed_spectrograms, labels=["Mixed"] * NUM_SAMPLES, save_dir=SAVE_DIR)

print(f"Dataset saved in {SAVE_DIR}")
