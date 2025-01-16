#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:41:42 2024

@author: jarin.ritu
"""
import os
import shutil

# File paths for train and test lists
train_list = "train.txt"
test_list = "test.txt"

# Path to the original DeepShip dataset
original_dataset_path = ""

# Destination directories for train and test sets
train_output_dir = "DeepShip_train"
test_output_dir = "DeepShip_test"

# Ensure the train and test directories exist
os.makedirs(train_output_dir, exist_ok=True)
os.makedirs(test_output_dir, exist_ok=True)

# Helper function to copy files to the appropriate directory
def copy_files(file_list, output_dir):
    with open(file_list, 'r') as f:
        lines = f.readlines()
        for line in lines:
            # Get the relative path from the original dataset
            relative_path = line.strip()  # Remove any surrounding whitespace/newlines
            # Construct full file path in the original dataset
            full_file_path = os.path.join(original_dataset_path, relative_path)
            # Construct the destination path
            destination_path = os.path.join(output_dir, relative_path)
            # Ensure the subdirectory exists
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            # Copy the file
            shutil.copy2(full_file_path, destination_path)
            print(f"Copied {full_file_path} to {destination_path}")

# Copy files for train set
copy_files(train_list, train_output_dir)

# Copy files for test set
copy_files(test_list, test_output_dir)

