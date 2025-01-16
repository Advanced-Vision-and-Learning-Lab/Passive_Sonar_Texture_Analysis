import torch.nn as nn
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
import pdb

class MelSpectrogramExtractor(nn.Module): 
    def __init__(self, sample_rate=16000, n_fft=512, win_length=512, hop_length=160, n_mels=64, fmin=50, fmax=8000):
        super(MelSpectrogramExtractor, self).__init__()
        
        # Settings for Spectrogram
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        
        self.spectrogram_extractor = Spectrogram(n_fft=win_length, hop_length=hop_length, 
                                                  win_length=win_length, window=window, center=center, 
                                                  pad_mode=pad_mode, 
                                                  freeze_parameters=True)

        ref = 1.0
        amin = 1e-10
        top_db = None
        
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=win_length, 
            n_mels=n_mels, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, freeze_parameters=True)
        
        t_n = n_mels

        self.bn0 = nn.BatchNorm2d(t_n)

    def forward(self, waveform):
        # pdb.set_trace()
        waveform = waveform.squeeze(1)
        spectrogram = self.spectrogram_extractor(waveform)
        log_mel_spectrogram = self.logmel_extractor(spectrogram)

        log_mel_spectrogram = log_mel_spectrogram.transpose(1, 3)
        log_mel_spectrogram = self.bn0(log_mel_spectrogram)
        log_mel_spectrogram = log_mel_spectrogram.transpose(1, 3)
            
        log_mel_spectrogram = log_mel_spectrogram.squeeze(1).transpose(1, 2)

        return log_mel_spectrogram


class Feature_Extraction_Layer(nn.Module):
    def __init__(self, input_feature='LogMelFBank', sample_rate=16000, window_length=4096, 
                 hop_length=1260, number_mels=64):
        super(Feature_Extraction_Layer, self).__init__()
        
        self.sample_rate = sample_rate   
        
        # Initialize log-mel filter bank
        win_length = window_length
        n_fft = window_length
        hop_length = hop_length 
        n_mels = number_mels
        fmin = 1
        fmax = 8000
        
        self.LogMelFBank = MelSpectrogramExtractor(
            sample_rate=sample_rate, 
            n_fft=n_fft,
            win_length=win_length, 
            hop_length=hop_length, 
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax
        )

    def forward(self, x):
        # pdb.set_trace()
        # Extract log-mel spectrogram from raw audio input (x)
        x = self.LogMelFBank(x)
        x = x.unsqueeze(1)  # Add channel dimension (for grayscale-like input)
        x = x.repeat(1, 1, 1, 1)
        return x