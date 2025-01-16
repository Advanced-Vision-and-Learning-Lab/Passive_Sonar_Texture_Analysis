import pickle
import numpy as np

# Load Task2GeneratorMatrices
with open("Task2GeneratorMatrices.pkl", "rb") as file:
    generator_matrices = pickle.load(file)

# Load your Task1 matrices
with open("Task1", "rb") as task1_file:
    task1_matrices = pickle.load(task1_file)

G1_in_task1 = task1_matrices["setting1"]
G2_in_task1 = task1_matrices["setting2"]

# Check if G1 and G2 are present in Task2GeneratorMatrices
G1_found = any(np.array_equal(value["GeneratorMatrix"], G1_in_task1) for value in generator_matrices.values())
G2_found = any(np.array_equal(value["GeneratorMatrix"], G2_in_task1) for value in generator_matrices.values())

if G1_found:
    print("G1 is found in Task2GeneratorMatrices.")
else:
    print("G1 is NOT found in Task2GeneratorMatrices.")

if G2_found:
    print("G2 is found in Task2GeneratorMatrices.")
else:
    print("G2 is NOT found in Task2GeneratorMatrices.")
