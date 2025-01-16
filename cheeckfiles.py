import os
import pandas as pd
import os
import numpy as np
import librosa

# def check_missing_files(root_dir, partition='train'):
#     """
#     Check if all files listed in the metadata.csv file exist in the dataset directory.

#     Args:
#         root_dir (str): Root directory of the dataset.
#         partition (str): Dataset partition to check ('train', 'test', 'validation').
    
#     Returns:
#         list: A list of missing files (if any).
#     """
#     # Path to the metadata file
#     metadata_file = os.path.join(root_dir, partition, 'metadata.csv')
    
#     # Check if the metadata file exists
#     if not os.path.exists(metadata_file):
#         raise FileNotFoundError(f"Metadata file {metadata_file} not found.")
    
#     # Read the metadata file
#     metadata = pd.read_csv(metadata_file)
    
#     # Directory containing the audio files
#     audio_dir = os.path.join(root_dir, partition, 'audio')

#     # List to store missing files
#     missing_files = []

#     # Iterate through the metadata
#     for _, row in metadata.iterrows():
#         # Construct the expected file path
#         audio_file_path = os.path.join(audio_dir, row['label'], f"{row['file_name']}.wav")
        
#         # Check if the file exists
#         if not os.path.exists(audio_file_path):
#             missing_files.append(audio_file_path)

#     # Print results
#     if missing_files:
#         print(f"Missing files in '{partition}' partition:")
#         for file in missing_files:
#             print(file)
#     else:
#         print(f"All files in the '{partition}' partition exist!")

#     return missing_files

# # Example usage
# root_dir = "Datasets/DeepShip/Segments_5s_32000Hz"
# missing_files = check_missing_files(root_dir, partition='validation')


import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import correlate

def check_similarity_cross_correlation(segment1_path, segment2_path):
    """
    Compare two audio segments using Cross-Correlation and other metrics.
    
    Args:
        segment1_path (str): Path to the first audio segment.
        segment2_path (str): Path to the second audio segment.
    
    Returns:
        dict: Similarity metrics including max cross-correlation, lag, correlation, and cosine similarity.
    """
    # Load the two audio segments
    signal1, sr1 = librosa.load(segment1_path, sr=None)
    signal2, sr2 = librosa.load(segment2_path, sr=None)
    
    # Ensure both signals have the same sampling rate
    if sr1 != sr2:
        raise ValueError("The segments have different sampling rates.")
    
    # Ensure both signals have the same length by trimming or padding
    min_length = min(len(signal1), len(signal2))
    signal1 = signal1[:min_length]
    signal2 = signal2[:min_length]
    
    # Compute cross-correlation
    cross_corr = correlate(signal1, signal2, mode='full')  # Cross-correlation
    lag = np.argmax(cross_corr) - len(signal1) + 1          # Lag at max cross-correlation
    max_cross_corr = np.max(cross_corr)                    # Maximum cross-correlation value

    # Normalize the cross-correlation
    normalized_cross_corr = max_cross_corr / (np.linalg.norm(signal1) * np.linalg.norm(signal2))

    # Compute other similarity metrics
    correlation = np.corrcoef(signal1, signal2)[0, 1]  # Correlation coefficient
    cosine_similarity = np.dot(signal1, signal2) / (np.linalg.norm(signal1) * np.linalg.norm(signal2))  # Cosine similarity

    return {
        "max_cross_correlation": normalized_cross_corr,
        "lag": lag,
        "correlation": correlation,
        "cosine_similarity": cosine_similarity
    }

# Example usage
train_segment_path = "Datasets/DeepShip/Segments_5s_32000Hz/train/audio/cargo/6.wav"
test_segment_path = "Datasets/DeepShip/Segments_5s_32000Hz/test/audio/cargo/6.wav"

similarity_metrics = check_similarity_cross_correlation(train_segment_path, test_segment_path)
print("Similarity metrics:")
for metric, value in similarity_metrics.items():
    print(f"{metric}: {value:.4f}")

# Decision based on Cross-Correlation
if similarity_metrics["max_cross_correlation"] > 0.9:  # Adjust threshold as needed
    print("The segments are likely from the same signal.")
else:
    print("The segments are likely from different signals.")


