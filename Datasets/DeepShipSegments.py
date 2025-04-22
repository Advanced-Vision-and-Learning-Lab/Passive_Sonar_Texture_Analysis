#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 17:14:01 2023
Define optimizer for training model
@author: jarin.ritu
"""
## Dataset object for the DeepShip Dataset
# import torchaudio
# import torch
# import os
# from torch.utils.data import Dataset
# from sklearn.model_selection import train_test_split

# class DeepShipSegments(Dataset):
#     def __init__(self, parent_folder, train_split=.8,val_test_split=.4,
#                   partition='train', random_seed= 42, shuffle = False, transform=None, 
#                   target_transform=None):
#         self.parent_folder = parent_folder
#         self.folder_lists = {
#             'train': [],
#             'test': [],
#             'val': []
#         }
#         self.train_split = train_split
#         self.val_test_split = val_test_split
#         self.partition = partition
#         self.transform = transform
#         self.shuffle = shuffle
#         self.target_transform = target_transform
#         self.random_seed = random_seed
#         self.class_mapping = {'Cargo': 0, 'Passengership': 1, 'Tanker': 2, 'Tug': 3}

#         # Loop over each label and subfolder
#         for label in ['Cargo', 'Passengership', 'Tanker', 'Tug']:
#             label_path = os.path.join(parent_folder, label)
#             subfolders = os.listdir(label_path)
            
#             # Split subfolders into training, testing, and validation sets
#             subfolders_train, subfolders_test_val = train_test_split(subfolders, 
#                                                                       train_size=train_split, 
#                                                                       shuffle=self.shuffle, 
#                                                                       random_state=self.random_seed)
#             subfolders_test, subfolders_val = train_test_split(subfolders_test_val, 
#                                                                 train_size=self.val_test_split, 
#                                                                 shuffle=self.shuffle, 
#                                                                 random_state=self.random_seed)

#             # Add subfolders to appropriate folder list
#             for subfolder in subfolders_train:
#                 subfolder_path = os.path.join(label_path, subfolder)
#                 self.folder_lists['train'].append((subfolder_path, self.class_mapping[label]))

#             for subfolder in subfolders_test:
#                 subfolder_path = os.path.join(label_path, subfolder)
#                 self.folder_lists['test'].append((subfolder_path, self.class_mapping[label]))

#             for subfolder in subfolders_val:
#                 subfolder_path = os.path.join(label_path, subfolder)
#                 self.folder_lists['val'].append((subfolder_path, self.class_mapping[label]))

#         self.segment_lists = {
#             'train': [],
#             'test': [],
#             'val': []
#         }

#         # Loop over each folder list and add corresponding files to file list
#         for split in ['train', 'test', 'val']:
#             for folder in self.folder_lists[split]:
#                 for root, dirs, files in os.walk(folder[0]):
#                     for file in files:
#                         if file.endswith('.wav'):
#                             file_path = os.path.join(root, file)
#                             label = folder[1]
#                             self.segment_lists[split].append((file_path, label))

#     def __len__(self):
#         return len(self.segment_lists[self.partition])

#     def __getitem__(self, idx):
#         file_path, label = self.segment_lists[self.partition][idx]
#         signal, sr = torchaudio.load(file_path, normalize = True)
#         label = torch.tensor(label)
#         if self.target_transform:
#             label = self.target_transform(label)

#         return signal, label, idx



## Dataset object for the Synthetic Datasets
import torchaudio
import torch
import os
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class DeepShipSegments(Dataset):
    def __init__(self, parent_folder, train_split=0.8, val_test_split=0.5,
                  partition='train', random_seed=42, shuffle=False,
                  transform=None, target_transform=None):
        
        self.parent_folder = parent_folder
        self.partition = partition
        self.transform = transform
        self.target_transform = target_transform
        self.random_seed = random_seed
        self.shuffle = shuffle
        self.class_mapping = {'Cargo': 0, 'Passengership': 1, 'Tanker': 2, 'Tug': 3}
        
        self.segment_lists = {
            'train': [],
            'val': [],
            'test': []
        }

        for class_name, label in self.class_mapping.items():
            class_path = os.path.join(parent_folder, class_name)
            all_files = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.endswith('.wav')]

            # Split into train and temp (val + test)
            train_files, temp_files = train_test_split(all_files, train_size=train_split,
                                                        shuffle=self.shuffle, random_state=self.random_seed)
            # Split temp into val and test
            val_files, test_files = train_test_split(temp_files, train_size=val_test_split,
                                                      shuffle=self.shuffle, random_state=self.random_seed)

            self.segment_lists['train'] += [(f, label) for f in train_files]
            self.segment_lists['val'] += [(f, label) for f in val_files]
            self.segment_lists['test'] += [(f, label) for f in test_files]

    def __len__(self):
        return len(self.segment_lists[self.partition])

    def __getitem__(self, idx):
        file_path, label = self.segment_lists[self.partition][idx]
        signal, sr = torchaudio.load(file_path, normalize=True)
        label = torch.tensor(label)

        if self.transform:
            signal = self.transform(signal)
        if self.target_transform:
            label = self.target_transform(label)

        return signal, label, idx

## Dataset object for the VTUAD Dataset
# import torchaudio
# import torch
# import os
# from torch.utils.data import Dataset   
# class DeepShipSegments(Dataset):
#     def __init__(self, root_dir, partition='train', 
#                   transform=None, target_transform=None):
#         """
#         Args:
#             root_dir (str): Path to the root merged dataset directory.
#             partition (str): One of 'train', 'val', 'test'.
#             transform (callable, optional): Optional transform to apply to the audio.
#             target_transform (callable, optional): Optional transform to apply to the label.
#         """
#         self.partition = partition
#         self.transform = transform
#         self.target_transform = target_transform
#         self.class_mapping = {
#                             'cargo': 0,
#                             'passengership': 1,
#                             'tanker': 2,
#                             'tug': 3,
#                             'background': 4
#                         }


#         base_path = os.path.join(root_dir, partition, 'audio')
#         self.segment_list = []

#         for class_name, label in self.class_mapping.items():
#             class_dir = os.path.join(base_path, class_name)
#             if not os.path.exists(class_dir):
#                 continue
#             for fname in os.listdir(class_dir):
#                 if fname.endswith('.wav'):
#                     file_path = os.path.join(class_dir, fname)
#                     self.segment_list.append((file_path, label))
#                     if len(self.segment_list) == 0:
#                         raise ValueError(f"No .wav files found in {base_path} for partition='{partition}'")


#     def __len__(self):
#         return len(self.segment_list)

#     def __getitem__(self, idx):
#         file_path, label = self.segment_list[idx]
#         signal, sr = torchaudio.load(file_path, normalize=True)
#         label = torch.tensor(label)

#         if self.transform:
#             signal = self.transform(signal)
#         if self.target_transform:
#             label = self.target_transform(label)
#         # print("signal.shape")
#         return signal, label, idx
