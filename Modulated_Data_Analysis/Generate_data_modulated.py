import numpy as np
from scipy.stats import kstwobign
import matplotlib.pyplot as plt

def generate_periodic_signal(freqs, length, sample_rate):
    t = np.linspace(0, length, int(sample_rate * length), endpoint=False)
    signal = np.zeros_like(t)
    for freq in freqs:
        amplitude_scaling = 1 / (freq ** 2) if freq > 100 else 1
        signal += np.sin(2 * np.pi * freq * t) * amplitude_scaling
    return signal

def generate_k_distribution(shape, size):
    # Generate a K-distributed signal with a given shape factor
    return kstwobign.rvs(size=size) * shape

def generate_rayleigh_distribution(scale, size):
    # Generate a Rayleigh-distributed signal
    return np.random.rayleigh(scale, size)


def create_ship_signal_V4(ship_type, total_length, sample_rate):
    segment_length = 3  # Length of each segment in seconds
    signal = np.array([])

    # Define frequency sets for structural variations
    freq_set_1 = np.logspace(np.log10(1), np.log10(50000), num=15)
    freq_set_2 = np.logspace(np.log10(1), np.log10(100000), num=15)

    # Explicitly define the combination for each ship type
    combinations = {
        'cargo': {'freqs': freq_set_1, 'dist': 'k'},
        'tanker': {'freqs': freq_set_1, 'dist': 'rayleigh'},
        'passenger': {'freqs': freq_set_2, 'dist': 'k'},
        'tug': {'freqs': freq_set_2, 'dist': 'rayleigh'}
    }

    combo = combinations[ship_type]
    freqs = combo['freqs']
    distribution_type = combo['dist']

    for _ in range(total_length // segment_length):
        segment_signal = generate_periodic_signal(freqs, segment_length, sample_rate)

        # Choose distribution parameters based on type
        if distribution_type == 'k':
            shape_params = [0.2, 0.6, 1.0, 1.4]  
            shape = np.random.choice(shape_params)  
            amplitude = generate_k_distribution(shape, len(segment_signal))
        else:  # Rayleigh distribution
            scale_params = [0.2, 0.6, 1.0, 1.4]  
            scale = np.random.choice(scale_params)  
            amplitude = generate_rayleigh_distribution(scale, len(segment_signal))

        signal = np.concatenate((signal, segment_signal * amplitude))

    return signal


ship_types = ['cargo', 'passenger', 'tug', 'tanker']  
length = 300 # Length of the audio in seconds
sample_rate = 16000  # Sampling rate
  
signals = {}
for ship_type in ship_types:
    signals[ship_type] = create_ship_signal_V4(ship_type, length, sample_rate)


import pickle
import os

directory = '../datasets_modulated'
file_path = os.path.join(directory, 'signals_modulated_V4.pkl')

if not os.path.exists(directory):
    os.makedirs(directory)

with open(file_path, 'wb') as fp:
    pickle.dump(signals, fp)
print('Data Generation Completed.')


import librosa
import librosa.display

def plot_and_save_audio_waveforms(signals, sample_rate, ship_types):
    for ship_type in ship_types:
        # Use the first 3 seconds of each signal as an example segment
        segment_duration = 3 
        start_sample = 0  # Starting at the beginning of the signal
        end_sample = int(segment_duration * sample_rate)
        
        # Extract the segment
        segment = signals[ship_type][start_sample:end_sample]
        
        # Plotting the waveform
        plt.figure(figsize=(6, 4))
        librosa.display.waveshow(segment, sr=sample_rate)
        plt.title(f'Audio Waveform for {ship_type.capitalize()} Ship')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        
        # Save the figure
        filename = f'../figures_modulated/{ship_type}_audio_waveform.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()  
        print(f'Saved waveform plot to {filename}')


figures_directory = '../figures_modulated'
if not os.path.exists(figures_directory):
    os.makedirs(figures_directory)
    
plot_and_save_audio_waveforms(signals, sample_rate, ship_types)
print('Audio Waveform Visualization and Saving Completed.')


from scipy.stats import rayleigh

# Parameters for the distributions
shape_params = [0.2, 0.6, 1.0, 1.4]
scale_params = [0.2, 0.6, 1.0, 1.4]

# Generate samples for K-distribution with different shape parameters
x_values = np.linspace(0, 5, 1000)  

# Plotting K-distribution with various shape parameters
plt.figure(figsize=(6, 4))
for shape in shape_params:
    y_values = kstwobign.pdf(x_values / shape) * shape  
    plt.plot(x_values, y_values, label=f'Shape = {shape}')

plt.title('K-Distribution with 4 Shape Parameters')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.savefig('../figures_modulated/k_distribution_shapes.png',dpi=300, bbox_inches='tight')
plt.close()

# Plotting Rayleigh distribution 
plt.figure(figsize=(6, 4))
for scale in scale_params:
    # Generate Rayleigh-distributed values
    y_values = rayleigh.pdf(x_values, scale=scale)
    plt.plot(x_values, y_values, label=f'Scale = {scale}')

plt.title('Rayleigh Distribution with 4 Scale Parameters')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.savefig('../figures_modulated/rayleigh_distribution_scales.png',dpi=300, bbox_inches='tight')
plt.close()


freq_set_cargo = np.logspace(np.log10(1), np.log10(50000), num=15)

# Choose a shape parameter for K-distribution from your specified options
shape_k_cargo = 1.0  

# Generate the periodic signal for "cargo"
segment_length_cargo = 3  # Segment length in seconds for the example
sample_rate_cargo = 16000  # Sample rate in Hz
segment_signal_cargo = generate_periodic_signal(freq_set_cargo, segment_length_cargo, sample_rate_cargo)

# Modulate the amplitude using the K-distribution with the chosen shape parameter
amplitude_modulation_cargo = generate_k_distribution(shape_k_cargo, segment_signal_cargo.size)

# Apply the modulation to the generated signal
modulated_signal_cargo = segment_signal_cargo * amplitude_modulation_cargo

# Plot the modulated signal for "cargo"
plt.figure(figsize=(6, 4))
plt.plot(np.linspace(0, segment_length_cargo, segment_signal_cargo.size), modulated_signal_cargo)
plt.title("Audio Waveform for 'Cargo' Ship with K-Distribution Modulation")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.savefig("../figures_modulated/cargo_example_segment_k_modulated.png",dpi=300, bbox_inches='tight')
plt.close()

# Print the frequency values used for the "cargo" segment
print("Frequency values for 'Cargo' segment (Hz):")
print(", ".join(f"{freq:.2f}" for freq in freq_set_cargo))
print(f"Shape used for K-distribution: {shape_k_cargo}")



