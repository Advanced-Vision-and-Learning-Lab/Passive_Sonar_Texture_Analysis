#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 17:15:03 2025

@author: jarin.ritu
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
def calculate_transition_entropy(signal, num_bins):
    differences = np.diff(signal)  # Calculate frame-to-frame differences
    hist, _ = np.histogram(differences, bins=num_bins, density=True)
    hist = hist + 1e-8  # Avoid log(0)
    hist = hist / np.sum(hist)  # Normalize
    entropy = -np.sum(hist * np.log(hist))
    return entropy


def compute_cumulative_entropy(signal, frame_size, hop_size, num_bins):
    """
    Compute the cumulative entropy for the signal with debug information.
    """
    num_frames = (len(signal) - frame_size) // hop_size + 1
    entropy_values = []

    for i in range(num_frames):


        frame = signal[i * hop_size : i * hop_size + frame_size]
        frame = frame / np.max(np.abs(frame))  # Normalize frame
        entropy = calculate_transition_entropy(frame, num_bins)
        entropy_values.append(entropy)

        # Debug: print entropy and normalized histogram for the first few frames
        if i < 10:
            print(f"Frame {i}: Entropy = {entropy}")
            hist, _ = np.histogram(frame, bins=num_bins, density=False)
            hist = hist + 1e-8
            hist = hist / np.sum(hist)
            print(f"Frame {i}: Histogram (sum = {np.sum(hist)}): {hist}")

    cumulative_entropy = np.cumsum(entropy_values)
    return cumulative_entropy



def visualize_entropy_curves(audio_path, frame_size, hop_size, num_bins):
    """
    Visualize direct and reverse cumulative entropy curves.
    """
    # Load audio
    signal, sr = librosa.load(audio_path, sr=None)
    print("Signal shape:", signal.shape)
    print("Signal snippet:", signal[:10])

    signal_reverse = signal[::-1]  # Reverse the signal

    # Compute cumulative entropy
    cumulative_entropy_direct = compute_cumulative_entropy(signal, frame_size, hop_size, num_bins)
    cumulative_entropy_reverse = compute_cumulative_entropy(signal_reverse, frame_size, hop_size, num_bins)
    print("Direct entropy (first 10 frames):", cumulative_entropy_direct[:10])
    print("Reverse entropy (first 10 frames):", cumulative_entropy_reverse[:10])

    
    plt.plot(signal)
    plt.title("Audio Signal")
    plt.show()
    
    signal_reverse = signal[::-1]
    plt.plot(signal[:100], label="Direct")
    plt.plot(signal_reverse[:100], label="Reverse")
    plt.legend()
    plt.show()


    # Plot cumulative entropy curves
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_entropy_direct, label="Direct Entropy", color="blue")
    plt.plot(cumulative_entropy_reverse, label="Reverse Entropy", color="red", linestyle="dashed")
    plt.xlabel("Frame Index")
    plt.ylabel("Cumulative Entropy")
    plt.title("Cumulative Entropy Curves (Direct vs Reverse)")
    plt.legend()
    plt.grid()
    plt.show()

# Parameters
audio_path = "1_Cargo-Segment_29.wav"  # Replace with your audio file path
frame_size = 512  # Reduce from 2048 to 256
hop_size = 128    # Reduce from 512 to 128


num_bins = 16      # Number of histogram bins

# Visualize entropy curves
visualize_entropy_curves(audio_path, frame_size, hop_size, num_bins)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# """
# Created on Sun Jan 19 18:30:00 2025

# @author: jarin.ritu
# """

# import numpy as np
# import librosa
# import matplotlib.pyplot as plt
# from scipy.signal import butter, filtfilt

# # Pre-Emphasis Function
# def pre_emphasize(signal, alpha=0.97):
#     """
#     Apply pre-emphasis filter to the signal.
#     Args:
#         signal (np.array): Input audio signal.
#         alpha (float): Pre-emphasis factor (default: 0.97).
#     Returns:
#         np.array: Pre-emphasized signal.
#     """
#     emphasized_signal = np.append(signal[0], signal[1:] - alpha * signal[:-1])
#     return emphasized_signal

# # Band-Pass Filter Function
# def bandpass_filter(signal, lowcut, highcut, sr, order=5):
#     """
#     Apply a band-pass filter to the signal.
#     Args:
#         signal (np.array): Input audio signal.
#         lowcut (float): Lower frequency cutoff (Hz).
#         highcut (float): Upper frequency cutoff (Hz).
#         sr (int): Sampling rate of the signal.
#         order (int): Filter order (default: 5).
#     Returns:
#         np.array: Filtered signal.
#     """
#     nyquist = 0.5 * sr
#     low = lowcut / nyquist
#     high = highcut / nyquist
#     b, a = butter(order, [low, high], btype='band')
#     filtered_signal = filtfilt(b, a, signal)
#     return filtered_signal

# # Preprocessing Function
# def preprocess_signal(audio_path, sr=None, pre_emphasis=True, band_pass=True, lowcut=300, highcut=3400):
#     """
#     Load and preprocess the audio signal with pre-emphasis and band-pass filtering.
#     Args:
#         audio_path (str): Path to the audio file.
#         sr (int): Sampling rate (default: None, original sampling rate).
#         pre_emphasis (bool): Whether to apply pre-emphasis.
#         band_pass (bool): Whether to apply band-pass filtering.
#         lowcut (float): Lower frequency cutoff for band-pass filtering (Hz).
#         highcut (float): Upper frequency cutoff for band-pass filtering (Hz).
#     Returns:
#         np.array: Preprocessed signal.
#         int: Sampling rate of the signal.
#     """
#     signal, sr = librosa.load(audio_path, sr=sr)
    
#     # Apply pre-emphasis
#     if pre_emphasis:
#         signal = pre_emphasize(signal)
    
#     # Apply band-pass filtering
#     if band_pass:
#         signal = bandpass_filter(signal, lowcut, highcut, sr)
    
#     return signal, sr

# # Entropy Calculation Function
# def calculate_transition_entropy(signal, num_bins):
#     differences = np.diff(signal)  # Calculate frame-to-frame differences
#     hist, _ = np.histogram(differences, bins=num_bins, density=True)
#     hist = hist + 1e-8  # Avoid log(0)
#     hist = hist / np.sum(hist)  # Normalize
#     entropy = -np.sum(hist * np.log(hist))
#     return entropy

# # Compute Cumulative Entropy Function
# def compute_cumulative_entropy(signal, frame_size, hop_size, num_bins):
#     """
#     Compute the cumulative entropy for the signal.
#     """
#     num_frames = (len(signal) - frame_size) // hop_size + 1
#     entropy_values = []

#     for i in range(num_frames):
#         frame = signal[i * hop_size : i * hop_size + frame_size]
#         entropy = calculate_transition_entropy(frame, num_bins)
#         entropy_values.append(entropy)

#         # Debug: Print entropy and histogram for the first few frames
#         if i < 10:
#             print(f"Frame {i}: Entropy = {entropy}")
#             hist, _ = np.histogram(frame, bins=num_bins, density=False)
#             hist = hist + 1e-8
#             hist = hist / np.sum(hist)
#             print(f"Frame {i}: Histogram (sum = {np.sum(hist)}): {hist}")

#     cumulative_entropy = np.cumsum(entropy_values)
#     return cumulative_entropy

# # Visualize Entropy Curves
# def visualize_entropy_curves(audio_path, frame_size, hop_size, num_bins):
#     """
#     Visualize direct and reverse cumulative entropy curves.
#     """
#     # Preprocess the signal
#     signal, sr = preprocess_signal(audio_path, sr=None, pre_emphasis=True, band_pass=True, lowcut=300, highcut=3400)
#     print("Signal shape:", signal.shape)
#     print("Signal snippet:", signal[:10])

#     signal_reverse = signal[::-1]  # Reverse the signal

#     # Compute cumulative entropy
#     cumulative_entropy_direct = compute_cumulative_entropy(signal, frame_size, hop_size, num_bins)
#     cumulative_entropy_reverse = compute_cumulative_entropy(signal_reverse, frame_size, hop_size, num_bins)
#     print("Direct entropy (first 10 frames):", cumulative_entropy_direct[:10])
#     print("Reverse entropy (first 10 frames):", cumulative_entropy_reverse[:10])

#     # Plot audio signal
#     plt.plot(signal)
#     plt.title("Audio Signal")
#     plt.show()

#     # Plot first 100 samples of direct and reverse signals
#     plt.plot(signal[:100], label="Direct")
#     plt.plot(signal_reverse[:100], label="Reverse")
#     plt.legend()
#     plt.show()

#     # Plot cumulative entropy curves
#     plt.figure(figsize=(10, 6))
#     plt.plot(cumulative_entropy_direct, label="Direct Entropy", color="blue")
#     plt.plot(cumulative_entropy_reverse, label="Reverse Entropy", color="red", linestyle="dashed")
#     plt.xlabel("Frame Index")
#     plt.ylabel("Cumulative Entropy")
#     plt.title("Cumulative Entropy Curves (Direct vs Reverse)")
#     plt.legend()
#     plt.grid()
#     plt.show()

# # Parameters
# audio_path = "acoustic-guitar-loop-f-91bpm-132687.mp3"  # Replace with your audio file path
# frame_size = 512  # Frame size for entropy computation
# hop_size = 256    # Hop size for frame shifting
# num_bins = 50     # Number of histogram bins

# # Visualize entropy curves
# visualize_entropy_curves(audio_path, frame_size, hop_size, num_bins)
