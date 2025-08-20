"""
Owned by Ryan Mehra. Licensed for free use.
Purpose: Example script for loading and inspecting RNA sequence datasets.
"""
import pandas as pd

d = pd.read_csv("./kaggle-data/new_training_sequences.csv")
print(d.shape)

# print d.sequence max length
print(d.sequence.str.len().min(), d.sequence.str.len().max())

d = pd.read_csv("./kaggle-data/new_validation_sequences.csv")
print(d.shape)

# print d.sequence max length
print(d.sequence.str.len().min(), d.sequence.str.len().max())
