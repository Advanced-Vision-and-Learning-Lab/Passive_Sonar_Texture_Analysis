#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 12:24:43 2024

@author: jarin.ritu
"""

import torch
import matplotlib.pyplot as plt
import os
from GAN import Generator, Discriminator
# Set parameters
latent_dim = 100  # Same latent dimension as used during training
batch_size = 1   # Number of samples to generate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create output directories
save_dir = "Generated_Spectrograms"
os.makedirs(save_dir, exist_ok=True)


# Number of classes in your dataset
num_classes = 1  # e.g., Cargo, Passengership, Tanker, Tug

# Instantiate the generator
generator = Generator(latent_dim=101, img_size=(126, 256)).to(device)
generator.eval()

# Generate spectrograms
z = torch.randn(batch_size, latent_dim).to(device)  # Noise vector
class_labels = torch.randint(0, num_classes, (batch_size,)).to(device)  # Random class labels
class_one_hot = torch.nn.functional.one_hot(class_labels, num_classes=num_classes).to(device)
z_conditional = torch.cat((z, class_one_hot.float()), dim=1)

generated_spectrograms = generator(z_conditional).detach().cpu()

# Save spectrograms
for i, spec in enumerate(generated_spectrograms):
    save_path = os.path.join(save_dir, f"generated_class_{class_labels[i].item()}_{i}.pt")
    torch.save(spec, save_path)
    print(f"Saved: {save_path} (Class {class_labels[i].item()})")
