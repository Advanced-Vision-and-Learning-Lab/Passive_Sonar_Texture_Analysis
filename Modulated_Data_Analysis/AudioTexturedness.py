import numpy as np
import librosa
import matplotlib.pyplot as plt

# Parameters
SAMPLE_RATE = 16000
DURATION = 5  # seconds
FRAME_SIZE = int(SAMPLE_RATE * 0.02)  # 20ms frames
HOP_SIZE = int(SAMPLE_RATE * 0.01)  # 10ms hop size
NUM_BINS = 50  # Number of amplitude levels for entropy computation

# Load and preprocess audio
def load_audio(file_path):
    signal, _ = librosa.load(file_path, sr=SAMPLE_RATE)
    target_length = SAMPLE_RATE * DURATION
    if len(signal) > target_length:
        signal = signal[:target_length]
    elif len(signal) < target_length:
        signal = np.pad(signal, (0, target_length - len(signal)))
    return signal

# Compute entropy for a single frame
def calculate_frame_entropy(frame, num_bins):
    histogram, _ = np.histogram(frame, bins=num_bins, density=True)
    histogram += 1e-8  # Avoid log(0)
    histogram /= np.sum(histogram)
    entropy = -np.sum(histogram * np.log(histogram))
    return entropy

# Compute cumulative entropy
def compute_cumulative_entropy(signal, frame_size, hop_size, num_bins):
    num_frames = (len(signal) - frame_size) // hop_size + 1
    frame_entropies = [
        calculate_frame_entropy(signal[i * hop_size : i * hop_size + frame_size], num_bins)
        for i in range(num_frames)
    ]
    cumulative_entropy = np.cumsum(frame_entropies)
    return cumulative_entropy

# Compute surface area S between direct and reverse entropy curves
def compute_surface_area(Hd, Hr, T_obs, H_max):
    S = T_obs * np.sum(np.abs(Hd - Hr)) / len(Hd)
    S_max = T_obs * H_max
    return S, S_max

# Plot cumulative entropy curves (recreate Figure 3)
def plot_entropy_curves_with_annotations(Hd_list, Hr_list, time_frames, labels, colors):
    plt.figure(figsize=(10, 6))

    for Hd, Hr, label, color in zip(Hd_list, Hr_list, labels, colors):
        plt.plot(time_frames, Hd, label=f"{label} (Direct)", color=color)
        plt.plot(time_frames, Hr, linestyle="dashed", label=f"{label} (Reverse)", color=color)
        
        # Highlight the surface area S for Speech
        if label == "Speech":
            plt.fill_between(time_frames, Hd, Hr, where=(Hd > Hr), interpolate=True, color=color, alpha=0.2, hatch='//')

    # Annotations for key metrics
    H_max = max(max(Hd_list), max(Hr_list))
    H_ref = np.log(NUM_BINS)  # Reference entropy for uniform white noise
    T_obs = max(time_frames)
    T_tex = T_obs * 0.75  # Example value
    H_Tobs = Hd_list[-1][-1]  # Example from Speech

    # Annotations for H_max, H_ref, H_Tobs, T_tex
    plt.axhline(H_max, linestyle="dotted", color="black", label="H_max")
    plt.axhline(H_ref, linestyle="solid", color="gray", label="H_ref")
    plt.axvline(T_tex, linestyle="dotted", color="red", label="T_tex")
    plt.scatter([T_obs], [H_Tobs], color="red", label="H_Tobs", zorder=5)

    # Axis labels and title
    plt.xlabel("Time (s)")
    plt.ylabel("Cumulative Temporal Entropy")
    plt.title("Cumulative Entropy Curves for Various Signals")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

# File paths and labels
# File paths and labels
audio_files = {
    "rain": "rain-110508.mp3",
    "typing machine": "typewriter-typing-68696.mp3",
    "speech": "speech-dramatic-female-38105.mp3"
}
colors = ["blue", "darkgreen", "red"]

# Process audio files
Hd_list, Hr_list = [], []
for file_path in audio_files.values():
    signal = load_audio(file_path)
    Hd = compute_cumulative_entropy(signal, FRAME_SIZE, HOP_SIZE, NUM_BINS)
    Hr = compute_cumulative_entropy(signal[::-1], FRAME_SIZE, HOP_SIZE, NUM_BINS)
    Hd_list.append(Hd)
    Hr_list.append(Hr)

# Generate time frames for plotting
time_frames = np.linspace(0, DURATION, len(Hd_list[0]))

# Generate the plot
plot_entropy_curves_with_annotations(Hd_list, Hr_list, time_frames, list(audio_files.keys()), colors)
