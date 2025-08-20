"""
Owned by Ryan Mehra. Licensed for free use.
Purpose: Converts and splits RNA sequence/label data for model training/validation.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# import os
# print(os.getcwd())
# # List all files in the current directory
# print(os.listdir())
# #List all files in the kaggle-data directory
# print("Files under:", os.listdir("./kaggle-data"))

# Load data
# try:
#     ext_ribonanza_sequences = pd.read_parquet("ext_ribonanza_sequences.parquet")
#     ext_ribonanza_sequences.to_csv("./kaggle-data/ext_ribonanza_sequences.csv", index=False)
#     print("ext_ribonanza_sequences.parquet converted to CSV")
#     print(ext_ribonanza_sequences.head())
#     print(ext_ribonanza_sequences.columns)
#     print(ext_ribonanza_sequences.shape)
# except FileNotFoundError:
#     print("ext_ribonanza_sequences.parquet not found")

# try:
#     ext_ribonanza_labels = pd.read_parquet("ext_ribonanza_labels.parquet")
#     ext_ribonanza_labels.to_csv("./kaggle-data/ext_ribonanza_labels.csv", index=False)
#     print("ext_ribonanza_labels.parquet converted to CSV")
#     print(ext_ribonanza_labels.head())
#     print(ext_ribonanza_labels.columns)
#     print(ext_ribonanza_labels.shape)
# except FileNotFoundError:
#     print("ext_ribonanza_labels.parquet not found")

# Load training data
training_sequences = pd.read_csv("./kaggle-data/train_sequences.csv")
training_labels = pd.read_csv("./kaggle-data/train_labels.csv")
training_sequences_v2 = pd.read_csv("./kaggle-data/train_sequences.v2.csv")
training_labels_v2 = pd.read_csv("./kaggle-data/train_labels.v2.csv")
ext_ribonanza_sequences = pd.read_csv("./kaggle-data/ext_ribonanza_sequences.csv")
ext_ribonanza_labels = pd.read_csv("./kaggle-data/ext_ribonanza_labels.csv")

# After loading training data, print initial shapes
print("Initial shapes:")
print("train_sequences:", training_sequences.shape)
print("train_labels:", training_labels.shape)
print("train_sequences.v2:", training_sequences_v2.shape)
print("train_labels.v2:", training_labels_v2.shape)
print("ext_ribonanza_sequences:", ext_ribonanza_sequences.shape)
print("ext_ribonanza_labels:", ext_ribonanza_labels.shape)

# Replace the assertion and mismatch checks with filtering based on matching target_id
def filter_pair(sequences, labels):
    # Add a new column 'target_id' to labels by stripping the trailing _<num> from 'ID'
    labels['target_id'] = labels['ID'].apply(lambda x: "_".join(x.split("_")[:-1]) if "_" in x else x)
    # Use the new 'target_id' column for matching.
    common_ids = set(sequences['target_id']).intersection(set(labels['target_id']))
    filtered_sequences = sequences[sequences['target_id'].isin(common_ids)].reset_index(drop=True)
    filtered_labels = labels[labels['target_id'].isin(common_ids)].reset_index(drop=True)
    return filtered_sequences, filtered_labels

training_sequences, training_labels = filter_pair(training_sequences, training_labels)
training_sequences_v2, training_labels_v2 = filter_pair(training_sequences_v2, training_labels_v2)
ext_ribonanza_sequences, ext_ribonanza_labels = filter_pair(ext_ribonanza_sequences, ext_ribonanza_labels)

print("\n\nAfter ensuring target_id match:")
print("train_sequences:", training_sequences.shape)
print("train_labels:", training_labels.shape)
print("train_sequences.v2:", training_sequences_v2.shape)
print("train_labels.v2:", training_labels_v2.shape)
print("ext_ribonanza_sequences:", ext_ribonanza_sequences.shape)
print("ext_ribonanza_labels:", ext_ribonanza_labels.shape)

# --- Begin filtering each training pair ---
# Filter sequences using the mask (the one-to-one filtering makes sense only for sequences)
mask1 = training_sequences['sequence'].str.fullmatch(r'[ACGU]+')
training_sequences = training_sequences.loc[mask1].reset_index(drop=True)
# For labels, select only records whose target_id exists in the filtered sequences.
training_labels = training_labels[training_labels['target_id'].isin(training_sequences['target_id'])].reset_index(drop=True)

mask2 = training_sequences_v2['sequence'].str.fullmatch(r'[ACGU]+')
training_sequences_v2 = training_sequences_v2.loc[mask2].reset_index(drop=True)
training_labels_v2 = training_labels_v2[training_labels_v2['target_id'].isin(training_sequences_v2['target_id'])].reset_index(drop=True)

mask3 = ext_ribonanza_sequences['sequence'].str.fullmatch(r'[ACGU]+')
ext_ribonanza_sequences = ext_ribonanza_sequences.loc[mask3].reset_index(drop=True)
ext_ribonanza_labels = ext_ribonanza_labels[ext_ribonanza_labels['target_id'].isin(ext_ribonanza_sequences['target_id'])].reset_index(drop=True)
# --- End filtering each training pair ---

print("\n\nAfter filtering for valid sequence characters:")
print("train_sequences:", training_sequences.shape)
print("train_labels:", training_labels.shape)
print("train_sequences.v2:", training_sequences_v2.shape)
print("train_labels.v2:", training_labels_v2.shape)
print("ext_ribonanza_sequences:", ext_ribonanza_sequences.shape)
print("ext_ribonanza_labels:", ext_ribonanza_labels.shape)

# Combine corresponding datasets while preserving row order
new_training_sequences = pd.concat([training_sequences, training_sequences_v2, ext_ribonanza_sequences], ignore_index=True)
new_training_labels = pd.concat([training_labels, training_labels_v2, ext_ribonanza_labels], ignore_index=True)
new_training_sequences.reset_index(drop=True, inplace=True)
new_training_labels.reset_index(drop=True, inplace=True)

print("\n\nCombined new training data shapes (before duplicate removal):")
print("new_training_sequences:", new_training_sequences.shape)
print("new_training_labels (1-to-many relationship):", new_training_labels.shape)

# Remove duplicate sequences and filter corresponding labels.
new_training_sequences = new_training_sequences.drop_duplicates(subset='sequence').reset_index(drop=True)
unique_target_ids = set(new_training_sequences['target_id'])
new_training_labels = new_training_labels[new_training_labels['target_id'].isin(unique_target_ids)].reset_index(drop=True)

print("\n\nAfter duplicate sequence removal:")
print("new_training_sequences:", new_training_sequences.shape)
print("new_training_labels (filtered):", new_training_labels.shape)

# Load validation data
validation_sequences = pd.read_csv("./kaggle-data/validation_sequences.csv")
validation_labels = pd.read_csv("./kaggle-data/validation_labels.csv")

# add target_id to validation_labels only
validation_labels['target_id'] = validation_labels['ID']\
    .apply(lambda x: "_".join(x.split("_")[:-1]) if "_" in x else x)

print("Validation data shapes:")
print("validation_sequences:", validation_sequences.shape)
print("validation_labels:", validation_labels.shape)

# Split 10% of new training data for validation using stratification on "target_id"
print("Splitting new training data for validation...")
test_size = 0.1
try:
    train_seq, val_seq = train_test_split(new_training_sequences, test_size=test_size, stratify=new_training_sequences["target_id"], random_state=42)
except ValueError as e:
    print(f"Error during stratified split: {e}. Proceeding without stratification.")
    train_seq, val_seq = train_test_split(new_training_sequences, test_size=test_size, random_state=42)

# Retrieve matching labels using the same target_id mapping.
def get_matching_labels(split_df, labels_df):
    valid_ids = set(split_df['target_id'])
    return labels_df[labels_df['target_id'].isin(valid_ids)].reset_index(drop=True)

val_labels = get_matching_labels(val_seq, new_training_labels)
train_labels_final = get_matching_labels(train_seq, new_training_labels)

print("\n\nData split completed.")
print("After splitting:")
print("Training split shape (sequences):", train_seq.shape)
print("Corresponding training labels shape:", train_labels_final.shape)
print("Validation split shape (sequences):", val_seq.shape)
print("Corresponding validation labels shape:", val_labels.shape)

# Append the new validation split to the existing validation data
new_validation_sequences = pd.concat([validation_sequences, val_seq], ignore_index=True)
new_validation_labels = pd.concat([validation_labels, val_labels], ignore_index=True)
print("\n\nCombined new training and validation data created.")
print("Final new validation data shapes:")
print("new_validation_sequences:", new_validation_sequences.shape)
print("new_validation_labels:", new_validation_labels.shape)

# assertions for target_id presence and non-null values
assert 'target_id' in train_labels_final.columns, "new_training_labels missing target_id"
assert train_labels_final['target_id'].notnull().all(), "Null target_id found in new_training_labels"
assert 'target_id' in new_validation_labels.columns, "new_validation_labels missing target_id"
assert new_validation_labels['target_id'].notnull().all(), "Null target_id found in new_validation_labels"

# Save the new training and validation data to CSV files
train_seq.to_csv("./kaggle-data/new_training_sequences.csv", index=False)
train_labels_final.to_csv("./kaggle-data/new_training_labels.csv", index=False)
new_validation_sequences.to_csv("./kaggle-data/new_validation_sequences.csv", index=False)
new_validation_labels.to_csv("./kaggle-data/new_validation_labels.csv", index=False)
print("New training and validation files saved.")
print("New files shapes:")
print("new_training_sequences.csv:", train_seq.shape)
print("new_training_labels.csv:", train_labels_final.shape)
print("new_validation_sequences.csv:", new_validation_sequences.shape)
print("new_validation_labels.csv:", new_validation_labels.shape)



