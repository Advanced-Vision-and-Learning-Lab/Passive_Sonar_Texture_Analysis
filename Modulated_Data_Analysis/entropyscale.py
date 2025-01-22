#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 18:12:04 2025

@author: jarin.ritu
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter



def pre_emphasis(signal, coeff=0.97):
    """
    Apply pre-emphasis filter to enhance high frequencies in the signal.
    """
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def bandpass_filter(signal, sr, lowcut=300, highcut=3400, order=4):
    """
    Apply band-pass filter to isolate relevant frequency components.
    """
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return lfilter(b, a, signal)


def calculate_entropy(signal, num_bins=50):
    """
    Calculate the entropy of a signal based on its histogram.
    """
    hist, _ = np.histogram(signal, bins=num_bins, density=True)
    hist = hist + 1e-8  # Avoid log(0)
    hist = hist / np.sum(hist)  # Normalize
    entropy = -np.sum(hist * np.log(hist))
    return entropy


def compute_cumulative_entropy(signal, frame_size, hop_size, num_bins=50):
    """
    Compute the cumulative entropy over overlapping frames.
    """
    num_frames = (len(signal) - frame_size) // hop_size + 1
    entropy_values = []

    for i in range(num_frames):
        frame = signal[i * hop_size : i * hop_size + frame_size]
        entropy = calculate_entropy(frame, num_bins)
        entropy_values.append(entropy)

    cumulative_entropy = np.cumsum(entropy_values)
    return cumulative_entropy, entropy_values


def map_entropy_to_scale(entropy_values, min_entropy, max_entropy):
    """
    Map entropy values to a scale of 0 to 5.
    """
    normalized_scores = (np.array(entropy_values) - min_entropy) / (max_entropy - min_entropy)
    texturedness_scores = normalized_scores * 5
    return np.clip(texturedness_scores, 0, 5)


def visualize_entropy_curves(audio_path, frame_size, hop_size, num_bins):
    """
    Visualize entropy curves and texturedness mapping.
    """
    # Load audio
    signal, sr = librosa.load(audio_path, sr=None)
    print(f"Loaded audio: {audio_path}, Sampling Rate: {sr}, Signal Length: {len(signal)}")

    # Preprocess signal
    signal = pre_emphasis(signal)
    signal = bandpass_filter(signal, sr)

    # Reverse signal
    signal_reverse = signal[::-1]

    # Compute cumulative entropy for direct and reverse signals
    cumulative_entropy_direct, entropy_values_direct = compute_cumulative_entropy(
        signal, frame_size, hop_size, num_bins
    )
    cumulative_entropy_reverse, entropy_values_reverse = compute_cumulative_entropy(
        signal_reverse, frame_size, hop_size, num_bins
    )

    # Map entropy to scale (assume min/max entropy range for scaling)
    min_entropy = min(min(entropy_values_direct), min(entropy_values_reverse))
    max_entropy = max(max(entropy_values_direct), max(entropy_values_reverse))
    texturedness_scores_direct = map_entropy_to_scale(entropy_values_direct, min_entropy, max_entropy)
    texturedness_scores_reverse = map_entropy_to_scale(entropy_values_reverse, min_entropy, max_entropy)

    print(f"Min Entropy: {min_entropy}, Max Entropy: {max_entropy}")
    print(f"First 10 Texturedness Scores (Direct): {texturedness_scores_direct[:10]}")
    print(f"First 10 Texturedness Scores (Reverse): {texturedness_scores_reverse[:10]}")

    # Plot signal
    plt.figure(figsize=(10, 4))
    plt.plot(signal)
    plt.title("Preprocessed Audio Signal")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()

    # Plot entropy curves
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_entropy_direct, label="Direct Entropy", color="blue")
    plt.plot(cumulative_entropy_reverse, label="Reverse Entropy", color="red", linestyle="dashed")
    plt.xlabel("Frame Index")
    plt.ylabel("Cumulative Entropy")
    plt.title("Cumulative Entropy Curve (Direct vs Reverse)")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot texturedness scale mapping
    plt.figure(figsize=(10, 6))
    plt.plot(texturedness_scores_direct, label="Texturedness Score (Direct)", color="green")
    plt.plot(texturedness_scores_reverse, label="Texturedness Score (Reverse)", color="orange", linestyle="dashed")
    plt.xlabel("Frame Index")
    plt.ylabel("Texturedness Scale")
    plt.title("Texturedness Score Curve (Direct vs Reverse)")
    plt.legend()
    plt.grid()
    plt.show()


# Parameters
audio_path = "0.wav"  # Replace with your audio file
frame_size = 512  # Frame size in samples
hop_size = 256    # Hop size in samples
num_bins = 50     # Number of bins for histogram

# Run visualization
visualize_entropy_curves(audio_path, frame_size, hop_size, num_bins)


# import numpy as np
# import librosa
# import matplotlib.pyplot as plt

# def compute_histogram_entropy(frame, num_bins):
#     hist, _ = np.histogram(frame, bins=num_bins, density=True)
#     hist = hist + 1e-8  # Avoid log(0)
#     hist = hist / np.sum(hist)  # Normalize
#     entropy = -np.sum(hist * np.log(hist))
#     return entropy

# def compute_cumulative_entropy(signal, frame_size, hop_size, num_bins):
#     num_frames = (len(signal) - frame_size) // hop_size + 1
#     entropy_values = []

#     for i in range(num_frames):
#         frame = signal[i * hop_size : i * hop_size + frame_size]
#         entropy = compute_histogram_entropy(frame, num_bins)
#         entropy_values.append(entropy)

#     cumulative_entropy = np.cumsum(entropy_values)
#     return cumulative_entropy, entropy_values

# def calculate_surface_area(Hd, Hr, Tabs, frame_size):
#     area = np.sum(np.abs(Hd - Hr)) * (Tabs / len(Hd))
#     return area

# def calculate_ftex(signal, frame_size, hop_size, num_bins, Tabs, K=5):
#     # Direct and reverse signals
#     reverse_signal = signal[::-1]

#     # Compute cumulative entropy curves
#     Hd, entropy_direct = compute_cumulative_entropy(signal, frame_size, hop_size, num_bins)
#     Hr, entropy_reverse = compute_cumulative_entropy(reverse_signal, frame_size, hop_size, num_bins)

#     # Calculate key parameters
#     Hmax = max(max(Hd), max(Hr))
#     S = calculate_surface_area(Hd, Hr, Tabs, frame_size)
#     Smax = Tabs * Hmax
#     Pr = S / Smax

#     Ttex = np.where(np.abs(Hd - Hr) < 1e-3)[0]  # Find convergence
#     Ttex = Ttex[0] if len(Ttex) > 0 else len(Hd)
#     PT = Ttex / len(Hd)

#     HTobs = Hd[-1]
#     Href = np.log(num_bins)
#     Prel = HTobs / Href

#     # Compute ftex
#     ftex = K * Prel * (1 - Pr * PT)
#     return ftex, Hd, Hr, S, Pr, PT, Prel

# def visualize_entropy_curves(Hd, Hr, ftex, S, Pr, PT, Prel):
#     plt.figure(figsize=(10, 6))
#     plt.plot(Hd, label="Direct Entropy", color="blue")
#     plt.plot(Hr, label="Reverse Entropy", color="red", linestyle="dashed")
#     plt.fill_between(range(len(Hd)), Hd, Hr, color="gray", alpha=0.3, label=f"Surface Area S = {S:.2f}")
#     plt.xlabel("Frame Index")
#     plt.ylabel("Cumulative Entropy")
#     plt.title(f"Cumulative Entropy Curves\nftex = {ftex:.2f}, Pr = {Pr:.2f}, PT = {PT:.2f}, Prel = {Prel:.2f}")
#     plt.legend()
#     plt.grid()
#     plt.show()

# # Parameters
# audio_path = "acoustic-guitar-loop-f-91bpm-132687.mp3"  # Replace with your audio file
# frame_size = 2048
# hop_size = 512
# num_bins = 50
# Tabs = 10  # Observation time in seconds

# # Load audio
# signal, sr = librosa.load(audio_path, sr=None)
# signal = signal[:int(sr * Tabs)]  # Trim to Tabs

# # Compute ftex
# ftex, Hd, Hr, S, Pr, PT, Prel = calculate_ftex(signal, frame_size, hop_size, num_bins, Tabs)

# # Visualize results
# visualize_entropy_curves(Hd, Hr, ftex, S, Pr, PT, Prel)

