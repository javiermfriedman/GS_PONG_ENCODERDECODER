# -------------------------------------------------------------
# model.py
#
# This file defines a fully connected (dense) autoencoder for 84x84 grayscale images.
# The model compresses the input image (flattened to 7056 features) down to a 3-dimensional
# latent (bottleneck) vector, then reconstructs the image from this compact representation.
# Such a model is useful for dimensionality reduction, visualization, or as a pre-processing
# step for downstream tasks. The encoder and decoder are both implemented as sequences of
# linear (fully connected) layers with ReLU activations, except for the output which uses Sigmoid.

# results: blurry images
# -------------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(84*84, 128), # Input: 7056-dim vector -> 128-dim (first compression)
            nn.ReLU(),
            nn.Linear(128, 64),    # 128-dim -> 64-dim (further compression)
            nn.ReLU(),
            nn.Linear(64, 12),     # 64-dim -> 12-dim (deeper compression)
            nn.ReLU(),
            nn.Linear(12, 3),      # 12-dim -> 3-dim (bottleneck/latent vector)
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 12),      # Bottleneck: 3-dim -> 12-dim (start reconstruction)
            nn.ReLU(),
            nn.Linear(12, 64),     # 12-dim -> 64-dim
            nn.ReLU(),
            nn.Linear(64, 128),    # 64-dim -> 128-dim
            nn.ReLU(),
            nn.Linear(128, 84*84), # 128-dim -> 7056-dim (reconstruct image)
            nn.Sigmoid(),          # Output in [0,1] for grayscale image
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded