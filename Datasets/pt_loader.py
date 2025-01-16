#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 11:53:20 2024

@author: jarin.ritu
"""

import torch
from torch.utils.data import Dataset, DataLoader
import os

import torchvision.transforms as transforms

resize_transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize spectrograms to 128x128
])

# Apply this transform when loading real spectrograms
class SpectrogramDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = []
        self.labels = []
        self.transform = transform
        self.classes = os.listdir(data_dir)

        for label, class_name in enumerate(self.classes):
            class_path = os.path.join(data_dir, class_name)
            for root, _, files in os.walk(class_path):
                for file in files:
                    if file.endswith(".pt"):
                        self.data.append(os.path.join(root, file))
                        self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        spectrogram = torch.load(self.data[idx])
        if self.transform:
            spectrogram = self.transform(spectrogram.unsqueeze(0))  # Add channel dimension
        label = self.labels[idx]
        return spectrogram, label

# Example usage
dataset = SpectrogramDataset(data_dir="DeepShip_Spectrograms", transform=resize_transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
