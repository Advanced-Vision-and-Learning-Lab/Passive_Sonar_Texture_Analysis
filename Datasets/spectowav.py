import torch
import torchaudio
import os
import matplotlib.pyplot as plt
from torchvision.transforms import Resize
import pdb

def spectrogram_to_audio(spectrogram, save_path, sample_rate=16000, n_fft=250, hop_length=64, win_length=250):
    """
    Converts a spectrogram back to audio using Griffin-Lim algorithm and saves it as a .wav file.
    """
    # Ensure the spectrogram has the correct frequency dimension
    if spectrogram.size(0) != n_fft // 2 + 1:
        resize_transform = Resize((n_fft // 2 + 1, spectrogram.size(1)))  # Resize frequency axis
        spectrogram = resize_transform(spectrogram.unsqueeze(0)).squeeze(0)
    # pdb.set_trace()
    spectrogram = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())


    # Griffin-Lim transform
    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,power=2, n_iter=128
    )
    print(f"Generated Spectrogram Shape: {spectrogram.shape}")
    print(f"Generated Spectrogram Stats: Min={spectrogram.min()}, Max={spectrogram.max()}, Mean={spectrogram.mean()}")


    
    # Convert spectrogram to waveform
    waveform = griffin_lim(spectrogram)
    print(f"Waveform Length: {waveform.size(-1)} samples")
    print(f"Duration: {waveform.size(-1) / 16000} seconds")

    # Save as .wav file
    torchaudio.save(save_path, waveform.unsqueeze(0), sample_rate)
    print(f"Saved audio to {save_path}")
    
    return waveform

# Directory where generated spectrograms are saved
generated_dir = "Generated_Spectrograms"
audio_save_dir = "Generated_Audio"
os.makedirs(audio_save_dir, exist_ok=True)

# Iterate through generated spectrograms
for file_name in os.listdir(generated_dir):
    if file_name.endswith(".pt"):
        # Load spectrogram
        spectrogram_path = os.path.join(generated_dir, file_name)
        spectrogram = torch.load(spectrogram_path).squeeze(0)  # Remove channel dimension
        
        # Convert to audio
        audio_path = os.path.join(audio_save_dir, file_name.replace(".pt", ".wav"))
        waveform = spectrogram_to_audio(spectrogram, audio_path)
        
        # Plot waveform for visualization (optional)
        plt.figure()
        plt.plot(waveform.numpy())
        plt.title(f"Waveform: {file_name}")
        plt.show()

# Load generated spectrogram
spectrogram = torch.load("DeepShip_Spectrograms/11_Cargo-Segment_1.pt")
# pdb.set_trace()
waveform = spectrogram_to_audio(spectrogram.squeeze(0), "DeepShip_Spectrograms/Segment_1.wav")

# Visualize the waveform
plt.plot(waveform.numpy().squeeze())
plt.title("Original Waveform")
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude")
plt.show()