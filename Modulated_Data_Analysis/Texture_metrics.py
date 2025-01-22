#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 08:50:56 2025

@author: jarin.ritu
"""

import os
import numpy as np
from scipy.stats import kurtosis, skew
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

SAVE_DIR = "./synthetic_dataset"  # Update if necessary



# Function to load spectrograms from SAVE_DIR
def load_spectrograms(save_dir, label):
    """
    Load spectrograms saved as .png files and handle RGBA channels.
    """
    files = [f for f in os.listdir(save_dir) if label in f and f.endswith(".png")]
    spectrograms = []
    for f in files:
        img = mpimg.imread(os.path.join(save_dir, f))
        if img.ndim == 3 and img.shape[2] == 4:  # RGBA data
            img = img[:, :, 0]  # Use only the first channel
        spectrograms.append(img)
    return spectrograms

# Load spectrograms
rayleigh_spectrograms = load_spectrograms(SAVE_DIR, "Rayleigh")
k_spectrograms = load_spectrograms(SAVE_DIR, "K-Distribution")
mixed_spectrograms = load_spectrograms(SAVE_DIR, "Mixed")

# Function to compute statistical texture metrics
def compute_statistical_metrics(signal):
    """
    Compute statistical metrics for a given signal or spectrogram.
    """
    return {
        "mean": np.mean(signal),
        "variance": np.var(signal),
        "skewness": skew(signal.flatten()),
        "kurtosis": kurtosis(signal.flatten())
    }

# Function to compute structural texture metrics
def compute_structural_metrics(spectrogram1, spectrogram2, win_size=3):
    """
    Compute structural similarity (SSIM) between two spectrograms.
    
    Parameters:
    - spectrogram1: First spectrogram (2D numpy array).
    - spectrogram2: Second spectrogram (2D numpy array).
    - win_size: Window size for SSIM (must be an odd number, <= smaller dimension of spectrograms).
    
    Returns:
    - Dictionary containing the SSIM index.
    """
    ssim_index = ssim(spectrogram1, spectrogram2, data_range=spectrogram1.max() - spectrogram1.min(), win_size=win_size)
    return {"ssim": ssim_index}


# Analyze Statistical Textures
def analyze_statistical_textures(spectrograms, label):
    """
    Analyze statistical textures for a set of spectrograms.
    Returns the metrics for further use.
    """
    stats_list = []
    print(f"\nAnalyzing Statistical Textures for {label} Spectrograms:")
    for i, spectrogram in enumerate(spectrograms):
        stats = compute_statistical_metrics(spectrogram)
        stats_list.append(stats)  # Store metrics for later use
        print(f"Spectrogram {i+1}: {stats}")
    return stats_list


# Analyze Structural Textures
def analyze_structural_textures(spectrograms, label):
    """
    Analyze structural textures for a set of spectrograms.
    Returns a list of SSIM metrics for consecutive pairs of spectrograms.
    """
    ssim_metrics_list = []  # List to store SSIM metrics
    print(f"\nAnalyzing Structural Textures for {label} Spectrograms:")
    
    # Compare SSIM between all pairs of spectrograms
    for i in range(len(spectrograms) - 1):
        ssim_metrics = compute_structural_metrics(spectrograms[i], spectrograms[i+1])
        ssim_metrics_list.append(ssim_metrics)  # Append metrics to the list
        print(f"SSIM between Spectrogram {i+1} and {i+2}: {ssim_metrics}")
    
    return ssim_metrics_list


# Run Analysis for Statistical Textures
# Analyze statistical textures and store metrics
rayleigh_stats = analyze_statistical_textures(rayleigh_spectrograms, "Rayleigh")
k_stats = analyze_statistical_textures(k_spectrograms, "K-Distribution")


# Run Analysis for Structural Textures
# Analyze structural textures
mixed_ssim = analyze_structural_textures(mixed_spectrograms, "Mixed")
# Check dimensions of the Mixed spectrograms
print(f"Mixed spectrogram dimensions: {[spectrogram.shape for spectrogram in mixed_spectrograms]}")


# Visualization Function
def plot_spectrogram(spectrogram, title):
    """
    Plot a spectrogram with proper normalization and scaling.
    
    Parameters:
    - spectrogram: 2D numpy array representing the spectrogram.
    - title: Title for the plot.
    """
    # Handle multichannel spectrograms (e.g., RGB-like data)
    if spectrogram.ndim == 3 and spectrogram.shape[2] == 4:
        spectrogram = spectrogram[:, :, 0]  # Use only one channel

    # Ensure spectrogram values are within [0, 1] for display
    spectrogram = np.clip(spectrogram, 0, None)  # Remove negative values
    normalized_spectrogram = (spectrogram - np.min(spectrogram)) / (np.max(spectrogram) - np.min(spectrogram))

    # Plot the spectrogram
    plt.figure(figsize=(6, 4))
    plt.imshow(normalized_spectrogram, aspect='auto', cmap='viridis')
    plt.title(title)
    plt.colorbar(label="Normalized Intensity")
    plt.show()


# Plot Example Spectrograms
plot_spectrogram(rayleigh_spectrograms[0], "Rayleigh Spectrogram Example")
plot_spectrogram(k_spectrograms[0], "K-Distribution Spectrogram Example")
plot_spectrogram(mixed_spectrograms[0], "Mixed Spectrogram Example")
import matplotlib.pyplot as plt

# Statistical metrics visualization
rayleigh_means = [stat['mean'] for stat in rayleigh_stats]
k_means = [stat['mean'] for stat in k_stats]

plt.boxplot([rayleigh_means, k_means], labels=["Rayleigh", "K-Distribution"])
plt.title("Comparison of Mean Intensity")
plt.ylabel("Mean Intensity")
plt.show()

# SSIM visualization
ssim_values = [ssim['ssim'] for ssim in mixed_ssim]
plt.plot(ssim_values, marker='o')
plt.title("SSIM Values for Mixed Spectrograms")
plt.xlabel("Spectrogram Pairs")
plt.ylabel("SSIM")
plt.show()
# Boxplot for mean intensity
plt.boxplot([rayleigh_means, k_means], tick_labels=["Rayleigh", "K-Distribution"])
plt.title("Comparison of Mean Intensity")
plt.ylabel("Mean Intensity")
plt.show()

# Boxplot for variance
rayleigh_variances = [stat['variance'] for stat in rayleigh_stats]
k_variances = [stat['variance'] for stat in k_stats]

plt.boxplot([rayleigh_variances, k_variances], tick_labels=["Rayleigh", "K-Distribution"])
plt.title("Comparison of Variance")
plt.ylabel("Variance")
plt.show()
