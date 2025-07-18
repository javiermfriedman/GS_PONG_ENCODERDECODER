"""
Script to count the number of files in the Autoencoder_data directory
"""

import os
import shutil

data_dir = "/Users/javierfriedman/Desktop/Research/EncoderDecoder_pong_greyscale/Autoencoder_data/reconstructed_images_greyscale"
out_dir = "/Users/javierfriedman/Desktop/Research/EncoderDecoder_pong_greyscale/Autoencoder_data/reconstructed_images_greyscale_reduced"

# Create output directory if it doesn't exist
os.makedirs(out_dir, exist_ok=True)

# List all files (not directories) in the source directory, sorted for consistency
files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
files.sort()

# Copy the first 25,000 files
for i, filename in enumerate(files[:25000]):
    src = os.path.join(data_dir, filename)
    dst = os.path.join(out_dir, filename)
    shutil.copy2(src, dst)
    if (i+1) % 1000 == 0:
        print(f"Copied {i+1} files...")

print(f"Copied {min(25000, len(files))} files from {data_dir} to {out_dir}.")




