import numpy as np
from scipy.stats import kstwobign
import pickle

# Generate statistical signals
def generate_statistical_signal(size, dist_type, param):
    if dist_type == 'k':
        return kstwobign.rvs(size=size) * param
    else:
        raise ValueError("Unsupported distribution type")

# Generate structural signals
def generate_structural_signal(freqs, length, sample_rate):
    t = np.linspace(0, length, int(sample_rate * length), endpoint=False)
    signal = np.sum([np.sin(2 * np.pi * f * t) for f in freqs], axis=0)
    return signal

# Combine statistical and structural signals
def create_hybrid_signal(stat_signal, struct_signal, weight_stat=0.5, weight_struct=0.5):
    return weight_stat * stat_signal + weight_struct * struct_signal

# Create dataset
def create_known_dataset(num_samples, sample_rate, length):
    dataset = []
    labels = []

    for _ in range(num_samples):
        # Generate statistical signal with random shape parameter
        stat_param = np.random.choice([0.5, 1.0, 1.5])
        stat_signal = generate_statistical_signal(size=sample_rate * length, dist_type='k', param=stat_param)

        # Generate structural signal with random frequency range
        freqs = np.logspace(np.log10(1), np.log10(500), num=10)
        struct_signal = generate_structural_signal(freqs, length, sample_rate)

        # Create hybrid signal
        hybrid_signal = create_hybrid_signal(stat_signal, struct_signal)

        # Extract labels
        labels.append({'stat_info': stat_param, 'struct_info': np.mean(freqs)})

        # Add to dataset
        dataset.append(hybrid_signal)

    return dataset, labels

# Save dataset
def save_dataset(dataset, labels, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump({'dataset': dataset, 'labels': labels}, f)
    print(f"Dataset saved to {file_path}")

# Parameters
num_samples = 100
sample_rate = 16000
length = 3  # in seconds
file_path = "known_dataset.pkl"

# Generate and save dataset
dataset, labels = create_known_dataset(num_samples, sample_rate, length)
save_dataset(dataset, labels, file_path)
