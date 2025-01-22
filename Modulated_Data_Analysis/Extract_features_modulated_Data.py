import numpy as np
import librosa
import scipy

def extract_statistical_features(signal):
    # Statistical features: mean, variance, skewness, zero-crossing rate, RMS energy
    mean = np.mean(signal)
    variance = np.var(signal)
    skewness = scipy.stats.skew(signal)
    zcr = np.mean(librosa.feature.zero_crossing_rate(signal + 0.0001)[0]) 
    rms = np.sqrt(np.mean(signal**2))
    return [mean, variance, skewness, zcr, rms]

def extract_frequency_features(signal, sample_rate):
    # Frequency-domain features: spectral centroid, spectral bandwidth, spectral flatness, spectral rolloff
    spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sample_rate)[0, 0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sample_rate)[0, 0]
    spectral_flatness = librosa.feature.spectral_flatness(y=signal)[0, 0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sample_rate)[0, 0]
    return [spectral_centroid, spectral_bandwidth, spectral_flatness, spectral_rolloff]


from skimage.feature import graycomatrix, graycoprops

def generate_mel_spectrogram(signal, sample_rate, n_mels=128):
    S = librosa.feature.melspectrogram(y=signal, sr=sample_rate, n_mels=n_mels)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB

def compute_texture_features(spectrogram, distances=[1], angles=[0], levels=256):
    max_val = spectrogram.max()
    if max_val == 0:
        max_val = 1  # Prevent division by zero

    spectrogram_normalized = (spectrogram - spectrogram.min()) / (max_val - spectrogram.min())
    spectrogram_quantized = (spectrogram_normalized * (levels - 1)).astype(int)
    glcm = graycomatrix(spectrogram_quantized, distances=distances, angles=angles, levels=levels)

    texture_features = {
        'contrast': graycoprops(glcm, 'contrast')[0, 0],
        'correlation': graycoprops(glcm, 'correlation')[0, 0],
        'energy': graycoprops(glcm, 'energy')[0, 0],
        'homogeneity': graycoprops(glcm, 'homogeneity')[0, 0]
    }
    return texture_features

def extract_mel_spectrogram_features(signal, sample_rate):
    mel_spectrogram = generate_mel_spectrogram(signal, sample_rate)
    glcm_features = compute_texture_features(mel_spectrogram)
    
    # Return the features as a list
    return [glcm_features['contrast'], glcm_features['correlation'], 
            glcm_features['energy'], glcm_features['homogeneity']]

ship_types = ['cargo', 'passenger', 'tug', 'tanker']  
import pickle
import pandas as pd

def segment_signal(signal, segment_length, sample_rate):
    num_samples_per_segment = segment_length * sample_rate
    return [signal[i:i + num_samples_per_segment] for i in range(0, len(signal), num_samples_per_segment)]


with open('../datasets_modulated/signals_modulated_V4.pkl', 'rb') as fp:
    signals = pickle.load(fp)
segment_length = 3  
sample_rate = 16000 
dataset = []

for ship_type in ship_types:
    segments = segment_signal(signals[ship_type], segment_length, sample_rate)
    for segment in segments:
        if len(segment) == segment_length * sample_rate:
            statistical_features = extract_statistical_features(segment)
            frequency_features = extract_frequency_features(segment, sample_rate)
            glcm_features = extract_mel_spectrogram_features(segment, sample_rate)
            dataset.append((statistical_features, frequency_features, glcm_features, ship_type))


df = pd.DataFrame(dataset, columns=['Statistical Features', 'Frequency Features', 'GLCM Features', 'Label'])
df.to_pickle('../datasets_modulated/features_segmented_dataframe_V4.pkl')

print("Segmentation and Features Extraction Done.")