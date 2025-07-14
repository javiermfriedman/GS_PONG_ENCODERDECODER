"""
This Autoencoder model is designed to be more effective at capturing fine-grained details 
in the 84x84 grayscale Pong game images compared to the previous version in model_2.

Key improvements include:
1.  Deeper Architecture: The network has more convolutional layers, allowing it to learn a more complex hierarchy of features.
2.  Batch Normalization: `BatchNorm2d` is applied after each convolutional layer. This stabilizes the learning process, accelerates training, and helps the model retain information about subtle features.
3.  Increased Feature Maps: The number of channels (filters) in the convolutional layers is significantly increased (from 16/32/64 to 32/64/128/256). This gives the model a higher capacity to learn distinct features for the ball, paddles, and score simultaneously.
4.  Strategic Pooling: Instead of aggressively downsampling the image after every convolution, this model uses a "detail-preserver" block—a convolutional layer without an immediate subsequent pooling layer. This allows the network to analyze features at a higher resolution (21x21) before the final downsampling, ensuring that small but critical game elements are not lost.
"""

import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, latent_dim=32):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # Input: (N, 1, 84, 84)
            
            # --- Block 1 ---
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2), # (N, 32, 84, 84)
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2), # (N, 32, 42, 42)
            
            # --- Block 2 ---
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # (N, 64, 42, 42)
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2), # (N, 64, 21, 21)
            
            # --- Block 3 (Detail-Preserver) ---
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # (N, 128, 21, 21)
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # No pooling here to preserve details
            
            # --- Block 4 ---
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), # (N, 256, 21, 21)
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2), # (N, 256, 10, 10)
            
            nn.Flatten(),
            nn.Linear(256 * 10 * 10, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 10 * 10),
            nn.ReLU(True),
            nn.Unflatten(1, (256, 10, 10)),
            
            # --- Block 1 ---
            # Using ConvTranspose2d to reverse the pooling and convolution
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=0), # (N, 128, 21, 21)
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # --- Block 2 (Reversing Detail-Preserver) ---
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1), # (N, 64, 21, 21)
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # --- Block 3 ---
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # (N, 32, 42, 42)
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            # --- Block 4 ---
            nn.ConvTranspose2d(32, 1, kernel_size=5, stride=2, padding=2, output_padding=1), # (N, 1, 84, 84)
            
            nn.Sigmoid() # To ensure pixel values are between 0 and 1
        )

    def forward(self, x):
        # The input x is expected to be flattened (N, 84*84)
        # Reshape it to (N, 1, 84, 84) for the convolutional layers
        x = x.view(-1, 1, 84, 84)
        
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        # Flatten the output to match the expected (N, 84*84) format
        decoded = decoded.view(-1, 84 * 84)
        return decoded

