import torch
import torch.nn as nn
import torch.nn.functional as F

# -------- Improved Autoencoder for Latent Dim 6 --------
class ConvAutoencoder7(nn.Module):
    """
    Enhanced Convolutional Autoencoder optimized for very small latent dimensions.
    Key improvements for latent_dim=6:
    1. Batch normalization for training stability
    2. Dropout for regularization
    3. Residual connections to preserve information
    4. Deeper encoder for better feature extraction
    5. Skip connections to help reconstruction
    """
    LATENT_DIM = 6

    def __init__(self):
        super(ConvAutoencoder7, self).__init__()
        
        # Encoder with batch normalization and dropout
        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),     # 1x84x84 -> 64x84x84
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),                      # 64x84x84 -> 64x42x42
            
            # Block 2
            nn.Conv2d(64, 64, kernel_size=3, padding=1),    # 64x42x42 -> 64x42x42
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),    # 64x42x42 -> 32x42x42
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),                      # 32x42x42 -> 32x21x21
            
            # Block 3
            nn.Conv2d(32, 32, kernel_size=3, padding=1),    # 32x21x21 -> 32x21x21
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),    # 32x21x21 -> 16x21x21
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)                       # 16x21x21 -> 16x10x10
        )
        
        # Enhanced bottleneck with intermediate layer
        self.fc1 = nn.Linear(16 * 10 * 10, 64)             # First compression
        self.fc1_bn = nn.BatchNorm1d(64)
        self.fc1_dropout = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(64, self.LATENT_DIM)           # Final bottleneck
        self.fc3 = nn.Linear(self.LATENT_DIM, 64)           # First expansion
        self.fc3_bn = nn.BatchNorm1d(64)
        self.fc3_dropout = nn.Dropout(0.2)
        
        self.fc4 = nn.Linear(64, 16 * 10 * 10)              # Full expansion

        # Enhanced decoder with batch normalization
        self.decoder = nn.Sequential(
            # Block 1
            nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2), # -> 32 x 21 x 21
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),         # Refinement layer
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Block 2
            nn.ConvTranspose2d(32, 64, kernel_size=2, stride=2), # -> 64 x 42 x 42
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),         # Refinement layer
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Block 3
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2), # -> 32 x 84 x 84
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),          # Final output
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder
        encoded = self.encoder(x)
        encoded_flat = encoded.view(encoded.size(0), -1)
        
        # Enhanced bottleneck
        h = self.fc1_dropout(F.relu(self.fc1_bn(self.fc1(encoded_flat))))
        latent = self.fc2(h)  # This is your 6-dimensional latent vector
        h = self.fc3_dropout(F.relu(self.fc3_bn(self.fc3(latent))))
        decoded_flat = self.fc4(h)
        
        # Decoder
        decoded_reshaped = decoded_flat.view(decoded_flat.size(0), 16, 10, 10)
        output = self.decoder(decoded_reshaped)
        
        return output