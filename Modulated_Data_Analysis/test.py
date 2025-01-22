import numpy as np
import librosa
import matplotlib.pyplot as plt

# Function to calculate entropy for each frame
def calculate_frame_entropy(frame, num_bins):
    histogram, _ = np.histogram(frame, bins=num_bins, density=True)
    histogram += 1e-8  # Avoid log(0)
    histogram /= np.sum(histogram)
    entropy = -np.sum(histogram * np.log(histogram))
    return entropy

# Compute cumulative entropy curves
def compute_cumulative_entropy(signal, frame_size, hop_size, num_bins):
    num_frames = (len(signal) - frame_size) // hop_size + 1
    frame_entropies = []

    for i in range(num_frames):
        frame = signal[i * hop_size : i * hop_size + frame_size]
        entropy = calculate_frame_entropy(frame, num_bins)
        frame_entropies.append(entropy)

    cumulative_entropy = np.cumsum(frame_entropies)
    return cumulative_entropy, frame_entropies

# Compute surface area S between direct and reverse curves
def compute_surface_area(Hd, Hr, Tabs, Hmax):
    S = Tabs * np.sum(np.abs(Hd - Hr))
    Smax = Tabs * Hmax
    return S, Smax

# Compute texturedness indicator
def compute_texturedness(S, Smax, Ttex, Tabs, HTobs, Href, Hmax):
    Pr = S / Smax  # Normalizing area ratio
    PT = Ttex / Tabs  # Texturedness time ratio
    Prel = HTobs / Href  # Normalizing final entropy ratio
    f_tex = 5 * Prel * (1 - Pr * (1 - PT))  # Scale to [0, 5]
    return f_tex, Pr, PT, Prel

# Main process
def process_audio_for_texturedness(audio_path, frame_size, hop_size, num_bins):
    # Load audio
    signal, sr = librosa.load(audio_path, sr=None)
    Tabs = len(signal) / sr  # Total observation time in seconds

    # Compute direct and reverse cumulative entropy
    Hd, frame_entropies_direct = compute_cumulative_entropy(signal, frame_size, hop_size, num_bins)
    Hr, frame_entropies_reverse = compute_cumulative_entropy(signal[::-1], frame_size, hop_size, num_bins)

    Hmax = max(np.max(Hd), np.max(Hr))
    HTobs = max(Hd[-1], Hr[-1])  # Final entropy value
    Href = np.log(num_bins)  # Reference entropy for uniform white noise

    # Compute surface area S
    S, Smax = compute_surface_area(Hd, Hr, Tabs, Hmax)

    # Assume Ttex = Tabs for simplicity (adjust if needed for better estimation)
    Ttex = Tabs

    # Compute texturedness indicator
    f_tex, Pr, PT, Prel = compute_texturedness(S, Smax, Ttex, Tabs, HTobs, Href, Hmax)

    # Plot cumulative entropy curves
    plt.figure(figsize=(10, 6))
    plt.plot(Hd, label="Direct Entropy", color="blue")
    plt.plot(Hr, label="Reverse Entropy", color="red", linestyle="dashed")
    plt.title(f"Cumulative Entropy Curves\n"
              f"f_tex = {f_tex:.2f}, Pr = {Pr:.2f}, PT = {PT:.2f}, Prel = {Prel:.2f}")
    plt.xlabel("Frame Index")
    plt.ylabel("Cumulative Entropy")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot texturedness score per frame
    texturedness_scores_direct = 5 * (1 - np.abs(frame_entropies_direct - Href) / Href)
    texturedness_scores_reverse = 5 * (1 - np.abs(frame_entropies_reverse - Href) / Href)

    plt.figure(figsize=(10, 6))
    plt.plot(texturedness_scores_direct, label="Texturedness Score (Direct)", color="green")
    plt.plot(texturedness_scores_reverse, label="Texturedness Score (Reverse)", color="orange", linestyle="dashed")
    plt.title("Texturedness Score Curve (Direct vs Reverse)")
    plt.xlabel("Frame Index")
    plt.ylabel("Texturedness Scale")
    plt.legend()
    plt.grid()
    plt.show()

# Parameters
audio_path = "non_textured_audio.wav"  # Replace with your audio file path
frame_size = 512
hop_size = 512
num_bins = 50

# Process the audio
process_audio_for_texturedness(audio_path, frame_size, hop_size, num_bins)
