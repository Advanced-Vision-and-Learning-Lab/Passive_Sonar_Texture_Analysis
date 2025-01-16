import os
import numpy as np
import librosa
from collections import defaultdict

# Define the paths to the folders
folders = {
    "train": "Segments_5s_32000Hz_2k/train/audio/cargo",
    "test": "Segments_5s_32000Hz_2k/test/audio/cargo",
    "val": "Segments_5s_32000Hz_2k/validation/audio/cargo"
}

# Collect file names and their locations
file_locations = defaultdict(list)
for folder_name, folder_path in folders.items():
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            file_locations[file_name].append(file_path)

# Find duplicate file names
duplicates = {file_name: paths for file_name, paths in file_locations.items() if len(paths) > 1}

# Function to calculate Mean Absolute Difference (MAD) between two audio files
def calculate_mad(file1, file2):
    # Load audio files
    y1, sr1 = librosa.load(file1, sr=None)
    y2, sr2 = librosa.load(file2, sr=None)

    # Ensure the sample rates match
    if sr1 != sr2:
        raise ValueError(f"Sample rates do not match for {file1} and {file2}")

    # Match lengths of audio files
    min_length = min(len(y1), len(y2))
    y1, y2 = y1[:min_length], y2[:min_length]

    # Calculate MAD
    mad = np.mean(np.abs(y1 - y2))
    return mad

# Check duplicates for content similarity
print(f"Total number of duplicate file names: {len(duplicates)}")
print("\nChecking duplicates for content similarity:")
for file_name, paths in duplicates.items():
    identical = True
    reference_path = paths[0]
    for path in paths[1:]:
        try:
            mad = calculate_mad(reference_path, path)
            if mad > 1e-6:  # Threshold to consider files identical (adjust if needed)
                identical = False
        except Exception as e:
            print(f"Error comparing {reference_path} with {path}: {e}")
            identical = False
    if identical:
        print(f"Duplicates for '{file_name}' are identical.")

