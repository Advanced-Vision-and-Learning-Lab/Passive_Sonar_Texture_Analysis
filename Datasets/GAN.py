#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 11:53:53 2024

@author: jarin.ritu
"""

import torch.nn as nn
import pdb
import torch

import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=101, img_size=(126, 256)):
        super(Generator, self).__init__()
        self.img_size = img_size  # Target spectrogram size
        self.init_size = (img_size[0], img_size[1] // 4)  # Frequency and reduced time dimension

        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size[0] * self.init_size[1]))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),  # Upsample time axis
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # Upsample to final time dimension
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.size(0), 128, self.init_size[0], self.init_size[1])  # Reshape to reduced size
        img = self.conv_blocks(out)
        img = img[:, :, :self.img_size[0], :self.img_size[1]]  # Ensure exact target size
        return img


class Discriminator(nn.Module):
    def __init__(self, img_size=(126, 256)):
        super(Discriminator, self).__init__()
        self.img_size = img_size

        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )

        # Compute flattened size dynamically
        dummy_input = torch.zeros(1, 1, *img_size)  # Batch size 1, 1 channel, img_size
        flattened_size = self._get_flattened_size(dummy_input)



        self.fc = nn.Sequential(
            nn.Linear(flattened_size, 1),  # Adjust based on flattened size
            nn.Sigmoid()
        )

    def _get_flattened_size(self, x):
        x = self.model[0](x)
        x = self.model[1](x)
        x = self.model[2](x)
        x = self.model[4](x)
        x = self.model[5](x)
        x = self.model[6](x)
        return x.view(x.size(0), -1).size(1)  # Flatten and get size

    def forward(self, img):
        features = self.model(img)
        flattened = features.view(features.size(0), -1)
        validity = self.fc(flattened)
        return validity
