#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 10:51:11 2024

@author: jarin.ritu
"""

import pickle

# Verify the Task3 file
with open("Task3", "rb") as file:
    data = pickle.load(file)

print("G*_1:\n", data["setting1"])
print("\nG*_2:\n", data["setting2"])
