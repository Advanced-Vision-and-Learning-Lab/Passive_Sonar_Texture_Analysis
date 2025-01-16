#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 10:10:25 2024

@author: jarin.ritu
"""

import os
import hashlib
import os
import hashlib

# Helper function to calculate the hash of a file
def calculate_file_hash(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

# Define the paths to the folders
folders = [
    "DeepShip/Segments_5s_32000Hz_2k/train/audio/cargo",
    "DeepShip/Segments_5s_32000Hz_2k/test/audio/cargo",
    "DeepShip/Segments_5s_32000Hz_2k/validation/audio/cargo"
]

# Collect file hashes and paths
file_hashes = {}
for folder in folders:
    for file_name in os.listdir(folder):
        file_path = os.path.join(folder, file_name)
        if os.path.isfile(file_path):  # Ensure it's a file
            file_hash = calculate_file_hash(file_path)
            if file_hash in file_hashes:
                file_hashes[file_hash].append(file_path)
            else:
                file_hashes[file_hash] = [file_path]

# Count duplicates
duplicate_files = {hash_val: paths for hash_val, paths in file_hashes.items() if len(paths) > 1}
total_duplicates = sum(len(paths) - 1 for paths in duplicate_files.values())  # Count extra files beyond the first one

# Print results
print(f"Total number of duplicated files: {total_duplicates}")
print("\nDuplicated files:")
for hash_val, paths in duplicate_files.items():
    print(f"Hash: {hash_val}")
    for path in paths:
        print(f" - {path}")
# import hashlib

# # Function to calculate the MD5 hash of a file
# def calculate_file_hash(filepath):
#     hasher = hashlib.md5()
#     with open(filepath, 'rb') as f:
#         while chunk := f.read(8192):
#             hasher.update(chunk)
#     return hasher.hexdigest()

# # Function to compare two files
# def compare_files(file1, file2):
#     hash1 = calculate_file_hash(file1)
#     hash2 = calculate_file_hash(file2)
    
#     print(f"Hash of {file1}: {hash1}")
#     print(f"Hash of {file2}: {hash2}")
    
#     if hash1 == hash2:
#         print("The files are identical.")
#     else:
#         print("The files are NOT identical.")

# # Example usage
# file1 = "Segments_5s_32000Hz_2k/validation/audio/cargo/1308.wav"
# file2 = "Segments_5s_32000Hz_2k/validation/audio/cargo/1108.wav"

# compare_files(file1, file2)
# import librosa
# import numpy as np

# def compare_audio_waveforms(file1, file2):
#     # Load the audio files
#     y1, sr1 = librosa.load(file1, sr=None)
#     y2, sr2 = librosa.load(file2, sr=None)

#     # Check sample rates
#     if sr1 != sr2:
#         print("The files are NOT identical (different sample rates).")
#         return

#     # Match lengths of audio data
#     min_length = min(len(y1), len(y2))
#     y1, y2 = y1[:min_length], y2[:min_length]

#     # Compute Mean Absolute Difference
#     mad = np.mean(np.abs(y1 - y2))
#     print(f"Mean Absolute Difference: {mad}")

#     if mad < 1e-6:  # Threshold for considering them identical
#         print("The files are identical (audio waveforms match).")
#     else:
#         print("The files are NOT identical (difference in audio waveforms).")
# # Example usage
# file1 = "Segments_5s_32000Hz_2k/validation/audio/cargo/1308.wav"
# file2 = "Segments_5s_32000Hz_2k/validation/audio/cargo/1108.wav"
# # Example usage
# compare_audio_waveforms(file1, file2)
import librosa
import librosa.display
import matplotlib.pyplot as plt

def plot_waveforms(file1, file2):
    y1, sr1 = librosa.load(file1, sr=None)
    y2, sr2 = librosa.load(file2, sr=None)

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(y1, sr=sr1, alpha=0.7)
    plt.title(f"Waveform of {file1}")

    plt.subplot(2, 1, 2)
    librosa.display.waveshow(y2, sr=sr2, alpha=0.7)
    plt.title(f"Waveform of {file2}")

    plt.tight_layout()
    plt.show()
# Example usage
file1 = "Segments_5s_32000Hz_2k/validation/audio/cargo/1308.wav"
file2 = "Segments_5s_32000Hz_2k/validation/audio/cargo/1108.wav"
# Example usage
plot_waveforms(file1, file2)
