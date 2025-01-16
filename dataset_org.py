#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 15:00:03 2024

@author: jarin.ritu
"""

import os
import shutil

# Define paths to your main dataset directory
dataset_dir = 'Datasets/VTUAD'
output_dir = 'Datasets/VTUAD_org'

# Define the splits and classes
splits = ['train', 'validation', 'test']
classes = ['background', 'cargo', 'passengership', 'tanker', 'tug']
inclusion_exclusion_folders = ['inclusion_2000_exclusion_4000', 'inclusion_3000_exclusion_5000', 'inclusion_4000_exclusion_6000']

# Create output directories if they don't exist
for split in splits:
    for class_name in classes:
        os.makedirs(os.path.join(output_dir, split, 'audio', class_name), exist_ok=True)

# Copy files from each inclusion-exclusion folder to the unified output directory
for inc_exc_folder in inclusion_exclusion_folders:
    for split in splits:
        for class_name in classes:
            source_dir = os.path.join(dataset_dir, inc_exc_folder, split, 'audio', class_name)
            target_dir = os.path.join(output_dir, split, 'audio', class_name)

            if os.path.exists(source_dir):
                for file_name in os.listdir(source_dir):
                    # Copy each file, avoiding overwriting by renaming if necessary
                    source_file = os.path.join(source_dir, file_name)
                    target_file = os.path.join(target_dir, file_name)

                    # If a file with the same name exists, add a unique suffix
                    if os.path.exists(target_file):
                        base_name, ext = os.path.splitext(file_name)
                        counter = 1
                        while os.path.exists(target_file):
                            target_file = os.path.join(target_dir, f"{base_name}_{counter}{ext}")
                            counter += 1

                    shutil.copy2(source_file, target_file)

print("All files have been successfully combined.")
