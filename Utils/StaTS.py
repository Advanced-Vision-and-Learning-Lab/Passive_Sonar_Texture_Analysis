import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.stats import entropy

def compute_entropy(signal, M=50):
    hist, _ = np.histogram(signal, bins=M, density=True)
    hist = hist + 1e-8  # avoid log(0)
    return entropy(hist)

def compute_cumulative_entropy(signal, frame_length, hop_length, M=50):
    n_frames = int(np.ceil((len(signal) - frame_length) / hop_length)) + 1
    Hd = []
    for i in range(1, n_frames + 1):
        end = i * hop_length
        frame = signal[:end] if end <= len(signal) else signal
        Hd.append(compute_entropy(frame, M))
    return np.array(Hd)

def compute_texturedness(Hd, Hr, Tobs, Tf, M=50, epsilon=0.01):
    nf = len(Hd)
    Hmax = max(np.max(Hd), np.max(Hr))
    S = np.sum(np.abs(Hd - Hr)) * Tf

    S_max = Tobs * Hmax
    P_S = S / S_max if S_max != 0 else 0

    # Estimate T_tex as earliest time where Hd and Hr converge
    for i in range(nf):
        if abs(Hd[i] - Hr[i]) < epsilon:
            T_tex = i * Tf
            break
    else:
        T_tex = Tobs  # if never converges

    P_T = T_tex / Tobs
    H_Tobs = Hd[-1]
    H_ref = np.log(M)
    P_rel = H_Tobs / H_ref

    I_tex = 5 * P_rel * (1 - P_T * P_S)
    return I_tex, S

def plot_hd_hr_shaded(signal, sr, label, color, M=150):
    import matplotlib.pyplot as plt
    import numpy as np

    Tobs = len(signal) / sr
    Tf = 0.02  # 20ms
    frame_length = int(Tf * sr)
    hop_length = frame_length

    Hd = compute_cumulative_entropy(signal, frame_length, hop_length, M)
    Hr = compute_cumulative_entropy(signal[::-1], frame_length, hop_length, M)
    time = np.arange(len(Hd)) * hop_length / sr

    # Plot entropy curves
    plt.plot(time, Hd, label=f"{label} Hd", color=color, linestyle='-')
    plt.plot(time, Hr, label=f"{label} Hr", color=color, linestyle='--')

    # Fill area between Hd and Hr
    plt.fill_between(time, Hd, Hr, color=color, alpha=0.2)

    # Horizontal line at log(M) = max entropy
    Hmax = np.log(M)
    plt.axhline(Hmax, color='gray', linestyle=':', linewidth=1)
    # plt.text(time[-1] + 0.1, Hmax, f"log(M) = {Hmax:.2f}", va='center', fontsize=8, color='gray')

    # Texturedness score
    I_tex, S = compute_texturedness(Hd, Hr, Tobs, Tf, M)
    H_Tobs = Hd[-1]
    print(f"Texturedness for {label}: I_tex = {I_tex:.3f}, Surface S = {S:.3f}, H_Tobs = {H_Tobs:.3f}")

    # Annotate score on plot
    plt.text(time[len(time)//2], max(Hd.max(), Hr.max()) + 0.1, f"I_tex = {I_tex:.2f}", 
             ha='center', color=color, fontsize=14, fontweight='bold')

    return I_tex

# Example synthetic signals
sr = 16000
duration = 5
t = np.linspace(0, duration, int(sr * duration), endpoint=False)

file_paths = {
    "White Noise": "pure_noise.wav",
    "Sine Wave": "pure_sine_1000Hz.wav",
    "Ship": "3_Cargo-Segment_3.wav"

}

colors = ['black', 'green', 'blue', 'darkgreen', 'red']

for (label, path), color in zip(file_paths.items(), colors):
    signal, sr = librosa.load(path, sr=16000, duration=5)
    I_tex = plot_hd_hr_shaded(signal, sr, label, color)

