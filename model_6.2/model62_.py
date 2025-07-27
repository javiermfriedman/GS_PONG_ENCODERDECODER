# model 6.2 architecture

"""
- **Increased Latent Dimension**: Bottleneck increased from 32 to 64.
- **Improved Decoder**: Uses larger kernels (4x4) and LeakyReLU for better upsampling.
- **Batch Normalization**: Applied after each convolutional layer to stabilize training.
- **Dropout**: Applied to the latent vector to discourage over-reliance on specific latent dimensions.
- **Transposed Convolutions**: Used for upsampling in the decoder.
"""

import torch
import torch.nn as nn

class ConvAutoencoderRegularized(nn.Module):
    # --- CHANGE 1: Increase latent dimension ---
    LATENT_DIM = 64  # Increased from 32

    def __init__(self):
        super(ConvAutoencoderRegularized, self).__init__()

        # Encoder (remains the same)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 1x84x84 -> 32x84x84
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # -> 32x42x42

            nn.Conv2d(32, 16, kernel_size=3, padding=1),  # -> 16x42x42
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # -> 16x21x21

            nn.Conv2d(16, 8, kernel_size=3, padding=1),   # -> 8x21x21
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)   # -> 8x10x10
        )

        # Bottleneck
        self.fc1 = nn.Linear(8 * 10 * 10, self.LATENT_DIM)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(self.LATENT_DIM, 8 * 10 * 10)

        # --- CHANGE 2: Improve the Decoder ---
        self.decoder = nn.Sequential(
            # Input: 8 x 10 x 10
            nn.ConvTranspose2d(8, 16, kernel_size=5, stride=2, padding=1), # -> 16x21x21
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),

            # Input: 16 x 21 x 21
            nn.ConvTranspose2d(16, 32, kernel_size=4, stride=2, padding=1), # -> 32x42x42
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # Input: 32 x 42 x 42
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # -> 1x84x84
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = x.view(x.size(0), 8, 10, 10)
        x = self.decoder(x)
        return x