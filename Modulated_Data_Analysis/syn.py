import pickle
import numpy as np
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import periodogram

# Load the dataset
def load_dataset(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data['dataset'], data['labels']

# Quantify statistical features
def extract_statistical_features(signal):
    return {
        'mean': np.mean(signal),
        'variance': np.var(signal),
        'skewness': skew(signal),
        'kurtosis': kurtosis(signal),
        'entropy': entropy(np.histogram(signal, bins=50)[0] + 1e-12)  # Add a small constant to avoid log(0)
    }

# Quantify structural features
def extract_structural_features(signal, sample_rate):
    freqs, power_spectrum = periodogram(signal, fs=sample_rate)
    return {
        'dominant_frequency': freqs[np.argmax(power_spectrum)],
        'spectral_entropy': entropy(power_spectrum + 1e-12),
        'autocorrelation': np.correlate(signal, signal, mode='full').max()
    }

# Analyze the dataset
def quantify_dataset(dataset, sample_rate):
    quantified_data = []
    for signal in dataset:
        stat_features = extract_statistical_features(signal)
        struct_features = extract_structural_features(signal, sample_rate)
        quantified_data.append({**stat_features, **struct_features})
    return quantified_data

# Parameters
file_path = "known_dataset.pkl"
sample_rate = 16000

# Load dataset and labels
dataset, labels = load_dataset(file_path)

# Quantify the dataset
quantified_data = quantify_dataset(dataset, sample_rate)

# # Display quantified information for the first few signals
# for i, (quantified, label) in enumerate(zip(quantified_data[:5], labels[:5])):
#     print(f"Signal {i + 1}:")
#     print(f"  Quantified Features: {quantified}")
#     print(f"  True Labels: {label}")
#     print()
import pandas as pd

# Convert quantified data to DataFrame
quantified_df = pd.DataFrame(quantified_data)

# Split into statistical and structural feature groups
stat_features = ['mean', 'variance', 'skewness', 'kurtosis', 'entropy']
struct_features = ['dominant_frequency', 'spectral_entropy', 'autocorrelation']

# Calculate variance
stat_variance = quantified_df[stat_features].var()
struct_variance = quantified_df[struct_features].var()

print("Variance of Statistical Features:")
print(stat_variance)
print("\nVariance of Structural Features:")
print(struct_variance)
