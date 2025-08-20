"""
Owned by Ryan Mehra. Licensed for free use.
Purpose: Downloads pre-trained model weights and datasets from KaggleHub.
"""
import kagglehub

# Download latest version
path = kagglehub.dataset_download("shujun717/ribonanzanet2d-final")

print("Path to dataset files:", path)

# Download latest version
path = kagglehub.dataset_download("shujun717/ribonanzanet-weights")

print("Path to dataset files:", path)