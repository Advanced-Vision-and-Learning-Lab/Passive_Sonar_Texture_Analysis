import torch.nn as nn
import torch
import torch.nn.functional as F
import torchaudio
from nnAudio import features
from Demo_Parameters import Parameters
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
import numpy as np
import pdb
import random
from librosa.util.exceptions import ParameterError
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1
        
    def forward(self, x):
        return torch.transpose(x, self.dim0, self.dim1)

class FeatureExtractionModel(torch.nn.Module):
    def __init__(self, sr):
        super(FeatureExtractionModel, self).__init__()
        
        # Parameters for the spectrogram
        imgsize = 460
        Fmax = sr / 2  # Nyquist frequency
        Nfft_lf = 32768  # Closest power of 2 above typical sample rates
        Nskip_lf = Nfft_lf // 2  # 80% overlap

        # Define the Spectrogram layer
        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=Nfft_lf,
            hop_length=Nskip_lf,
            power=2,
            return_complex=False
        ).cuda()  # Move to GPU if available
    
    def forward(self, signal):
        # pdb.set_trace()
        # Apply spectrogram transform to the input signal
        spectrogram = self.spectrogram(signal).unsqueeze(1)
        return spectrogram 
class Feature_Extraction_Layer(nn.Module):
    def __init__(self, input_feature, window_length, window_size, hop_size, mel_bins, fmin, fmax, classes_num,
                 hop_length, sample_rate=8000, RGB=False, downsampling_factor=2, frame_shift=10.0):
        super(Feature_Extraction_Layer, self).__init__()

        # Convert window and hop length to ms
        window_length /= 1000
        hop_length /= 1000
        window = 'hann'
        center = False
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        self.frame_shift = frame_shift  
        self.bn = nn.BatchNorm2d(101)
        # pdb.set_trace()
    


        if RGB:
            num_channels = 3
            MFCC_padding = nn.ZeroPad2d((3, 6, 16, 16))
        else:
            num_channels = 1
            MFCC_padding = nn.ZeroPad2d((1, 0, 4, 0))
        
        
        self.num_channels = num_channels
        self.input_feature = input_feature

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # Spec augmenter
        # self.spec_augmenter = SpecAugmentation(time_drop_width=48, time_stripes_num=2, 
        #     freq_drop_width=8, freq_stripes_num=2)

        # Return Mel Spectrogram that is 48 x 48
        self.Log_Mel_Spectrogram = nn.Sequential(self.spectrogram_extractor,
                                                self.logmel_extractor,
                                                Transpose(1, 3),
                                                self.bn,
                                                Transpose(1, 3))

        #self.downsample = nn.AvgPool2d(kernel_size=(1, downsampling_factor))

    #     # # Return Mel Spectrogram that is 48 x 48  MITLL new
        self.Mel_Spectrogram = nn.Sequential(
        features.mel.MelSpectrogram(
            sample_rate,
            n_mels=1024,  # 128 Mel bins
            win_length=8192,  # Window length of 8192
            hop_length=1024,  # Hop length of 1024
            n_fft=8192,  # Set n_fft equal to win_length for STFT
            verbose=False,
            fmax=sample_rate / 4 
        ),
        nn.ZeroPad2d((1, 4, 0, 4))  # Optional padding
    )
    #     self.Mel_Spectrogram = nn.Sequential(
    #     features.mel.MelSpectrogram(
    #         sample_rate,
    #         n_mels=64,  # 128 Mel bins
    #         win_length=1024,  # Window length of 8192
    #         hop_length=512,  # Hop length of 1024
    #         n_fft=1024,  # Set n_fft equal to win_length for STFT
    #         verbose=False

    #     ),
    #     nn.ZeroPad2d((1, 4, 0, 4))  # Optional padding
    # )
        # # Return MFCC that is 16 x 48 (TDNN models) or 48 x 48 (CNNs)
        # self.MFCC = nn.Sequential(features.mel.MFCC(sr=sample_rate, n_mfcc=16,
        #                                             n_fft=int(
        #                                                 window_length*sample_rate),
        #                                             win_length=int(
        #                                                 window_length*sample_rate),
        #                                             hop_length=int(
        #                                                 hop_length*sample_rate),
        #                                             n_mels=48, center=False, verbose=False), MFCC_padding)

        # # Return STFT that is 48 x 48
        self.STFT = nn.Sequential(features.STFT(sr=sample_rate,n_fft=int(window_length*sample_rate), 
                                        hop_length=int(hop_length*sample_rate),
                                        win_length=int(window_length*sample_rate), 
                                        output_format='Magnitude',
                                        freq_bins=48,verbose=False), nn.ZeroPad2d((1,0,0,0)))
        
        # # Return STFT that is 48 x 48
        # Adjust n_fft to be smaller than the input length
        # self.STFT = nn.Sequential(features.STFT(sr=sample_rate, 
        #                                         n_fft=int(window_length*sample_rate), 
        #                                         hop_length=96,
        #                                         win_length=256,
        #                                         output_format='Magnitude',
        #                                         freq_bins=129, verbose=False), 
        #                           nn.ZeroPad2d((0, 0, 0, 0)))
        self.STFT2 = nn.Sequential(features.STFT(sr=sample_rate, 
                                                n_fft=int(window_length*sample_rate), 
                                                hop_length=256,
                                                win_length=256,
                                                output_format='Magnitude',
                                                freq_bins=129, verbose=False), 
                                  nn.ZeroPad2d((20, 1, 0, 0)))
        # Return GFCC that is 64 x 48
        self.GFCC = nn.Sequential(features.Gammatonegram(sr=sample_rate,
                                                         hop_length=int(
                                                             hop_length*sample_rate),
                                                         n_fft=int(
                                                             window_length*sample_rate),
                                                         verbose=False, n_bins=64), nn.ZeroPad2d((1, 0, 0, 0)))

        # Return CQT that is 64 x 48
        self.CQT = nn.Sequential(features.CQT(sr=sample_rate,
                                              hop_length=int(hop_length*sample_rate), n_bins=48,
                                               verbose=False), nn.ZeroPad2d((0, 0, 0, 0)))

        # Return VQT that is 64 x 48
        self.VQT = nn.Sequential(features.VQT(sr=sample_rate, hop_length=int(hop_length*sample_rate),
                                              n_bins=48, earlydownsample=False, verbose=False), nn.ZeroPad2d((0, 5, 0, 0)))
        
        # Initialize new FBankLayer

        self.features = {'Log_Mel_Spectrogram':self.Log_Mel_Spectrogram,'Mel_Spectrogram':self.Mel_Spectrogram,
                         'STFT': self.STFT, 'GFCC': self.GFCC,
                         'CQT': self.CQT, 'VQT': self.VQT}


    
    def forward(self, x):
        # pdb.set_trace()
        x = self.features[self.input_feature](x)
        # pdb.set_trace()
        x = x.repeat(1, self.num_channels, 1, 1)
        # if torch.isnan(x).any():
        #     raise ValueError(f"NaN values found in signal from file {x}")
        #     pdb.set_trace()
        # x = torch.log1p(x)
        # if self.training:
        #     x = self.spec_augmenter(x)
        # pdb.set_trace()

        # if torch.isnan(x).any():
        #     raise ValueError(f"NaN values found in signal from file {x}")
        #     pdb.set_trace()
        return x
    
    # def forward(self, x):
    #     # pdb.set_trace()
    #     # Apply first STFT
    #     x = self.features[self.input_feature](x).unsqueeze(1)
        
    #     # Extract magnitude and phase
    #     magnitude = x[..., 0]
    #     phase = x[..., 1]
    
    #     # Convert to complex tensor
    #     complex_tensor = magnitude * torch.exp(1j * phase)
    
    #     # Extract real and imaginary parts
    #     real_part = torch.real(complex_tensor)
    #     imag_part = torch.imag(complex_tensor)
    
    #     # # Padding the signal to fit n_fft requirements
    #     padding_size = max(0, (self.features[self.input_feature][0].n_fft // 2) - real_part.shape[-1])
    #     real_part_padded = F.pad(real_part, (padding_size, padding_size))
    #     imag_part_padded = F.pad(imag_part, (padding_size, padding_size))
    
    #     # Combine channels (optional step if needed to reduce channels)
    #     real_part_padded = real_part_padded.mean(dim=1, keepdim=True)
    #     imag_part_padded = imag_part_padded.mean(dim=1, keepdim=True)
    
    #     # Apply second STFT on both real and imaginary parts
    #     real_stft = self.STFT2(real_part_padded).unsqueeze(1)
    #     imag_stft = self.STFT2(imag_part_padded).unsqueeze(1)
    #     # pdb.set_trace()
    
    #     # Combine real and imaginary part features
    #     combined_features = torch.cat([real_stft, imag_stft], dim=1)
    #     combined_features = combined_features.mean(dim=1, keepdim=True)
    
    #     return combined_features



    # def forward(self, x):
    #     pdb.set_trace()
        
    #     # Apply the first STFT with hop_length=96
    #     self.STFT1 = nn.Sequential(features.STFT(sr=4096, 
    #                                              n_fft=512, 
    #                                              hop_length=96,
    #                                              win_length=256,
    #                                              output_format='MagnitudePhase',  # Use Magnitude and Phase
    #                                              freq_bins=84, verbose=False),
    #                                nn.ZeroPad2d((1, 0, 0, 0))).to(device)
        
    #     # Apply the first STFT
    #     x = self.STFT1(x)  
        
    #     # Extract magnitude and phase from STFT output
    #     magnitude = x[..., 0]  # Magnitude is in the first channel
    #     phase = x[..., 1]      # Phase is in the second channel
        
    #     # Convert magnitude and phase to a complex tensor
    #     complex_tensor = magnitude * torch.exp(1j * phase)
        
    #     # Extract real and imaginary parts from the complex tensor
    #     real_part = torch.real(complex_tensor)
    #     imag_part = torch.imag(complex_tensor)
    
    #     # Apply the second STFT with hop_length=256 on real and imaginary parts
    #     self.STFT2 = nn.Sequential(features.STFT(sr=4096, 
    #                                              n_fft=512,  # Set to reasonable size
    #                                              hop_length=256,
    #                                              win_length=256,
    #                                              output_format='Complex',
    #                                              freq_bins=84, verbose=False),
    #                                nn.ZeroPad2d((1, 0, 0, 0))).to(device)
        
    #     # Apply the second STFT to the real and imaginary parts
    #     real_stft = self.STFT2(real_part)  # Second STFT on the real part
    #     imag_stft = self.STFT2(imag_part)  # Second STFT on the imaginary part
    
    #     # Combine the results
    #     combined_features = torch.cat([real_stft, imag_stft], dim=1)
        
    #     return combined_features
    
# import matplotlib.pyplot as plt
# plt.figure(figsize=(20, 12))  # Width = 10, Height = 8

# # Visualize the first feature map
# plt.imshow(x[0, 0].cpu().detach().numpy(), cmap='viridis')
# plt.colorbar()

# # Set axis labels
# plt.xlabel("Time (frames)")
# plt.ylabel("Frequency (mel-bins)")

# # Set title
# plt.title("Feature Map VTUAD")

# plt.show()