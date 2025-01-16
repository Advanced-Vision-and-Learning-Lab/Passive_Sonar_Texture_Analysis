#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 09:51:01 2024

@author: jarin.ritu
"""

import os
import shutil

# Define paths
source_base_dir = 'Datasets/DeepShip'  # Base directory of your dataset
destination_base_dir = 'Datasets/DeepShip_Split'  # Where you want to store train/test/val splits
split_file_path = 'split_indices.txt'  # Path to your split txt file

# Create directories for train, val, and test
os.makedirs(os.path.join(destination_base_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(destination_base_dir, 'val'), exist_ok=True)
os.makedirs(os.path.join(destination_base_dir, 'test'), exist_ok=True)

# Helper function to copy files based on split type
def copy_files(split, file_paths):
    split_dir = os.path.join(destination_base_dir, split)
    for path in file_paths:
        class_folder = path.split('/')[-3]  # Extracts class (e.g., Cargo, Tanker)
        target_dir = os.path.join(split_dir, class_folder)
        os.makedirs(target_dir, exist_ok=True)
        file_name = os.path.basename(path)
        destination = os.path.join(target_dir, file_name)
        
        # Copy the file to the appropriate directory
        shutil.copy2(path, destination)

# Read and parse the txt file
with open(split_file_path, 'r') as file:
    current_split = None
    file_paths = []

    for line in file:
        line = line.strip()
        
        # Identify which split section we're in
        if 'Train indices and paths:' in line:
            if current_split and file_paths:
                copy_files(current_split, file_paths)
            current_split = 'train'
            file_paths = []
        elif 'Validation indices and paths:' in line:
            if current_split and file_paths:
                copy_files(current_split, file_paths)
            current_split = 'val'
            file_paths = []
        elif 'Test indices and paths:' in line:
            if current_split and file_paths:
                copy_files(current_split, file_paths)
            current_split = 'test'
            file_paths = []
        elif line:  # Process paths
            file_path = line.split(': ', 1)[1]  # Get path after index
            file_paths.append(file_path)

    # Copy any remaining files for the last split
    if current_split and file_paths:
        copy_files(current_split, file_paths)

print("Files have been copied to their respective train, val, and test folders.")
