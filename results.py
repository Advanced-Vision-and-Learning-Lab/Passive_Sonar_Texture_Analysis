#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 10:16:10 2024

@author: jarin.ritu
"""

import librosa
import librosa.display
import matplotlib.pyplot as plt

import librosa
import matplotlib.pyplot as plt
import numpy as np

import librosa
import matplotlib.pyplot as plt
import numpy as np

def plot_audio_segments(audio_file, segment_duration=5):
    # Load the audio file
    y, sr = librosa.load(audio_file, sr=None)  # y is the waveform, sr is the sample rate
    
    # Calculate the number of samples for the given segment duration
    segment_length = segment_duration * sr
    
    # Determine the number of segments
    num_segments = int(np.ceil(len(y) / segment_length))
    
    # Create a plot for each segment
    for i in range(num_segments):
        start_sample = int(i * segment_length)
        end_sample = int(min((i + 1) * segment_length, len(y)))
        segment = y[start_sample:end_sample]

        # Create a time array for the x-axis of the segment
        time = np.linspace(start_sample / sr, end_sample / sr, len(segment))

        # Plot the audio waveform for each segment
        plt.figure(figsize=(10, 4))
        plt.plot(time, segment, color='blue', alpha=0.7)
        plt.title(f'Waveform of Segment {i + 1}', fontsize=14)
        plt.xlabel('Time (seconds)', fontsize=12)
        plt.ylabel('Amplitude', fontsize=12)
        
        plt.grid(False)  # Add a grid for better readability
        plt.tight_layout()
        plt.show()

# Replace 'your_audio_file.wav' with the path to your audio file
plot_audio_segments('1.wav', segment_duration=5)

